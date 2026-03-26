import torch.nn as nn 
import torch 
from .model_utils import Linear,SinusoidalPosEmb
import numpy as np
import torch.nn.functional as F

class AdaptiveWeighting(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.adaptive_weight = nn.Sequential(
            SinusoidalPosEmb(dim=hidden_dim),
            Linear(in_features=hidden_dim, out_features=1,init_bias=0,init_weight=np.sqrt(1/3)),
        )
    def forward(self,t):
        return self.adaptive_weight(t).squeeze(-1)
    

class Model(nn.Module):
    def __init__(self,
        input_dim,                        # Number of color channels at input.
        output_dim,                       # Number of color channels at output.
        condition_dim   = 0,            # Number of class labels, 0 = unconditional.
        hidden_dim      = 256,              # latent dimension for noise and class labels embedding
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim

        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))

        # Mapping 
        emb_dim = hidden_dim
        self.t_step_encoder = nn.Sequential(
            # PositionalEmbedding(num_channels=emb_dim,max_positions=1000),
            SinusoidalPosEmb(dim=emb_dim),
            Linear(in_features=emb_dim, out_features=emb_dim * 2, **init),
            nn.SiLU(),
            Linear(in_features=emb_dim * 2, out_features=emb_dim , **init))
        if condition_dim > 0:
            self.condition_encoder = nn.Sequential(
            Linear(in_features=condition_dim, out_features=emb_dim * 2, **init),
            nn.SiLU(),
            Linear(in_features=emb_dim * 2, out_features=emb_dim, **init))
        else:
            self.condition_encoder = None

        # Model 

        self.lin1 = Linear(in_features = input_dim, out_features = hidden_dim, **init)
        self.lin2 = Linear(in_features = hidden_dim, out_features = hidden_dim, **init)
        self.lin3 = Linear(in_features = hidden_dim, out_features = hidden_dim, **init)
        self.lin4 = Linear(in_features = hidden_dim, out_features = output_dim, **init)

    def forward(self, x , t, condition = None):

        emb = self.t_step_encoder(t)
        if (self.condition_dim is not None) and( condition is not None):
            tmp = condition 
            emb = emb + self.condition_encoder(tmp)
        # encoder 
        x = F.silu(self.lin1(x) + emb)
        x = F.silu(self.lin2(x) + emb)
        x = F.silu(self.lin3(x) + emb)
        x = self.lin4(x) 
        return x


class JointConditionModel(nn.Module):
    def __init__(self,
        input_dim,                        
        condition1_dim   = 0,            # Number of class labels, 0 = unconditional.
        condition2_dim   = 0,
        hidden_dim      = 256,              # latent dimension for noise and class labels embedding
        share_condition_encoder = False    ## remove this parameter 
    ):
        super().__init__()
        assert condition1_dim > 0 and condition2_dim > 0, "condition dim should be greater than 0"
        self.input_dim = input_dim 
        self.condition1_dim = condition1_dim
        self.condition2_dim = condition2_dim
        self.hidden_dim = hidden_dim

        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))

        # Mapping 
        emb_dim = hidden_dim
        self.t_step_encoder = nn.Sequential(
            # PositionalEmbedding(num_channels=emb_dim,max_positions=1000),
            SinusoidalPosEmb(dim=emb_dim),
            Linear(in_features=emb_dim, out_features=emb_dim *2, **init),
            nn.SiLU(),
            Linear(in_features=emb_dim * 2, out_features=emb_dim , **init))
        
        condition_emb_dim = emb_dim
        self.condition1_encoder = nn.Sequential(
                Linear(in_features=condition1_dim, out_features=emb_dim , **init),
                nn.SiLU(),
                Linear(in_features=emb_dim , out_features=condition_emb_dim, **init))
        self.condition2_encoder = nn.Sequential(
                Linear(in_features=condition2_dim, out_features=emb_dim , **init),
                nn.SiLU(),
                Linear(in_features=emb_dim, out_features=condition_emb_dim, **init))
        print("######################## condition1_encoder and condition2_encoder are different ########################")
        self.condition_emb_dim = condition_emb_dim
        self.condition_encoder = nn.Sequential(
            Linear(in_features=condition_emb_dim, out_features=emb_dim, **init),
            nn.SiLU(),
        )
        # Model 
        self.lin1 = Linear(in_features = input_dim, out_features = hidden_dim, **init)  
        self.layers = nn.ModuleList([
            Linear(in_features = hidden_dim, out_features = hidden_dim, **init),
            Linear(in_features = hidden_dim, out_features = hidden_dim, **init),
            Linear(in_features = hidden_dim, out_features = hidden_dim, **init),
        ])
        self.output = Linear(in_features = hidden_dim, out_features = input_dim, **init)

    def forward(self, x,t, condition1 = None,condition2 = None):

        emb = self.t_step_encoder(t)
        if condition1 is not None:
            condition1_emb = self.condition1_encoder(condition1)
        else:
            condition1_emb = torch.zeros_like(emb)
        
        if condition2 is not None:
            condition2_emb = self.condition2_encoder(condition2)
        else:
            condition2_emb = torch.zeros_like(emb)

        condition_emb = condition1_emb + condition2_emb
        emb = emb + self.condition_encoder(condition_emb)

        # encoder 
        x = F.silu(self.lin1(x) + emb)
        for layer in self.layers:
            x = F.silu(layer(x) + emb)
        x = self.output(x)
        
        return x


class ForwardModel(nn.Module):
    def __init__(self,
        state_dim,
        action_dim,
        next_state_dim,
        hidden_dim = 256,
        deterministic = False,
        action_range = None): ### action range is not used in this implementation 
        super().__init__()
        self.deterministic = deterministic
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.next_state_dim = next_state_dim
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        self.action_lin = Linear(in_features = action_dim, out_features = hidden_dim, **init)
        self.state_lin = Linear(in_features = state_dim, out_features = hidden_dim, **init)
        self.next_state_layers = nn.Sequential(
            Linear(in_features = hidden_dim*2, out_features = hidden_dim*2, **init),
            nn.SiLU(),
            Linear(in_features = hidden_dim*2, out_features = hidden_dim*2, **init),
            nn.SiLU(),
            Linear(in_features = hidden_dim*2, out_features = next_state_dim, **init)
        )
        self.estimated_norm_layers = nn.Sequential(
            Linear(in_features = hidden_dim, out_features = hidden_dim*2, **init),
            nn.SiLU(),  
            Linear(in_features = hidden_dim*2, out_features = 1, **init),
            nn.ReLU()
        )

    def forward(self,state,action):
        h_s = self.state_lin(state)
        h_a = self.action_lin(action)
        h = F.silu(torch.cat([h_s,h_a],dim=-1))
        next_state = self.next_state_layers(h)      

        estimated_norm = self.estimated_norm_layers(h_s)

        norm = torch.norm(next_state - state,dim=-1,keepdim=True)
        direction = (next_state - state) / (norm + 1e-6)

        if self.deterministic:
            return direction, None ,norm, estimated_norm
        else:
            raise NotImplementedError
    def predict(self,state,action,deterministic = True):
        mean,logstd, norm, estimated_norm = self.forward(state,action)
        if deterministic:
            next_state = mean * norm + state
            return mean, norm, next_state,estimated_norm
        else:
            raise NotImplementedError
    
    
    def compute_loss(self,state,action,next_state):
        info = {}
        delta_direction = next_state - state
        delta_direction_norm = torch.norm(delta_direction,dim=-1,keepdim=True)
        normalized_direction = delta_direction / (delta_direction_norm + 1e-6)

        mean,logstd, norm,estimated_norm = self.forward(state,action)
        norm_loss = (norm - delta_direction_norm).pow(2).mean(-1)
        estimated_norm_loss = (estimated_norm - delta_direction_norm).pow(2).mean(-1)

        info['norm_loss'] = norm_loss.mean().item()
        info['estimated_norm_loss'] = estimated_norm_loss.mean().item()

        forward_loss = (mean * norm + state - next_state).pow(2).mean(-1)
        info['forward_loss'] = forward_loss.mean().item()

        if self.deterministic:
            direction_loss = (mean - normalized_direction).pow(2).mean(-1)
            loss = estimated_norm_loss + forward_loss
            info['direction_loss'] = direction_loss.mean().item()
            
            
        else:
            raise NotImplementedError

        return loss,info

# class JointConditionModel(nn.Module):
#     def __init__(self,
#         input_dim,                        
#         condition1_dim   = 0,            # Number of class labels, 0 = unconditional.
#         condition2_dim   = 0,
#         hidden_dim      = 256,              # latent dimension for noise and class labels embedding
#         share_condition_encoder = True
#     ):
#         super().__init__()
#         assert condition1_dim > 0 and condition2_dim > 0, "condition dim should be greater than 0"
#         self.input_dim = input_dim 
#         self.condition1_dim = condition1_dim
#         self.condition2_dim = condition2_dim
#         self.hidden_dim = hidden_dim

#         init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))

#         # Mapping 
#         emb_dim = hidden_dim
#         self.t_step_encoder = nn.Sequential(
#             # PositionalEmbedding(num_channels=emb_dim,max_positions=1000),
#             SinusoidalPosEmb(dim=emb_dim),
#             Linear(in_features=emb_dim, out_features=emb_dim *2, **init),
#             nn.SiLU(),
#             Linear(in_features=emb_dim * 2, out_features=emb_dim , **init))
        
#         condition_emb_dim = emb_dim//2
#         if share_condition_encoder and (condition1_dim == condition2_dim):
#             self.condition1_encoder = nn.Sequential(
#                 Linear(in_features=condition1_dim, out_features=emb_dim, **init),
#                 nn.SiLU(),
#                 Linear(in_features=emb_dim, out_features=condition_emb_dim, **init))
#             self.condition2_encoder = self.condition1_encoder## I'm not sure if this is correct
#         else:
#             self.condition1_encoder = nn.Sequential(
#                 Linear(in_features=condition1_dim, out_features=emb_dim , **init),
#                 nn.SiLU(),
#                 Linear(in_features=emb_dim , out_features=condition_emb_dim, **init))
#             self.condition2_encoder = nn.Sequential(
#                 Linear(in_features=condition2_dim, out_features=emb_dim , **init),
#                 nn.SiLU(),
#                 Linear(in_features=emb_dim, out_features=condition_emb_dim, **init))
#             print("######################## condition1_encoder and condition2_encoder are different ########################")
#         self.condition_emb_dim = condition_emb_dim
#         # Model 
#         self.lin1 = Linear(in_features = input_dim, out_features = hidden_dim, **init)  
#         self.layers = nn.ModuleList([
#             Linear(in_features = hidden_dim, out_features = hidden_dim, **init),
#             Linear(in_features = hidden_dim, out_features = hidden_dim, **init),
#         ])
#         self.output = Linear(in_features = hidden_dim, out_features = input_dim, **init)

#     def forward(self, x,t, condition1 = None,condition2 = None):

#         emb = self.t_step_encoder(t)
#         if condition1 is not None:
#             condition1_emb = self.condition1_encoder(condition1)
#         else:
#             condition1_emb = torch.zeros_like(emb)[:,:self.condition_emb_dim]
        
#         if condition2 is not None:
#             condition2_emb = self.condition2_encoder(condition2)
#         else:
#             condition2_emb = torch.zeros_like(emb)[:,:self.condition_emb_dim]
#         condition_emb = torch.cat([condition1_emb,condition2_emb],dim=-1)
#         emb = emb + condition_emb

#         # encoder 
#         x = F.silu(self.lin1(x) + emb)
#         for layer in self.layers:
#             x = F.silu(layer(x) + emb)
#         x = self.output(x)
#         return x


# class ForwardModel(nn.Module):
#     def __init__(self,
#         state_dim,
#         action_dim,
#         next_state_dim,
#         hidden_dim = 256,
#         deterministic = False,
#         action_range = None): ### action range is not used in this implementation 
#         super().__init__()
#         self.deterministic = deterministic
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.next_state_dim = next_state_dim
#         init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
#         self.action_lin = Linear(in_features = action_dim, out_features = hidden_dim, **init)
#         self.state_lin = Linear(in_features = state_dim, out_features = hidden_dim, **init)
#         self.delta_direction_layers = nn.Sequential(
#             Linear(in_features = hidden_dim*2, out_features = hidden_dim*2, **init),
#             nn.SiLU(),
#             Linear(in_features = hidden_dim*2, out_features = hidden_dim*2, **init),
#             nn.SiLU(),
#         )
#         self.estimated_norm_layers = nn.Sequential(
#             Linear(in_features = hidden_dim, out_features = hidden_dim*2, **init),
#             nn.SiLU(),  
#         )
#         self.norm_layers = nn.Sequential(
#             Linear(in_features = hidden_dim * 2, out_features = hidden_dim*2, **init),
#             nn.SiLU(),  
#         )
#         if deterministic:
#             self.delta_direction_output_lin = Linear(in_features = hidden_dim*2, out_features = next_state_dim, **init)
#         else:
#             self.delta_direction_outpu_mean = Linear(in_features = hidden_dim*2, out_features = next_state_dim, **init)
#             self.delta_direction_outpu_logstd = Linear(in_features = hidden_dim*2, out_features = next_state_dim, **init)
        
#         self.estimated_norm_output_lin = nn.Sequential(
#                 Linear(in_features = hidden_dim*2, out_features = 1, **init),
#                 nn.ReLU())
#         self.norm_output_lin = nn.Sequential(
#                 Linear(in_features = hidden_dim*2, out_features = 1, **init),
#                 nn.ReLU())

#     def forward(self,state,action):
#         h_s = self.state_lin(state)
#         h_a = self.action_lin(action)
#         h_direction = F.silu(torch.cat([h_s,h_a],dim=-1))
#         h_direction = self.delta_direction_layers(h_direction) 
#         h_norm = self.norm_layers(h_direction)
        
#         h_estimated_norm = self.estimated_norm_layers(h_s)

#         if self.deterministic:
#             return self.delta_direction_output_lin(h_direction), None ,self.norm_output_lin(h_norm), self.estimated_norm_output_lin(h_estimated_norm)
#         else:
#             return self.delta_direction_outpu_mean(h_direction),self.delta_direction_outpu_logstd(h_direction), self.norm_output_lin(h_norm), self.estimated_norm_output_lin(h_estimated_norm)
        
#     def predict(self,state,action,deterministic = True):
#         mean,logstd, norm, estimated_norm = self.forward(state,action)
#         if deterministic:
#             next_state = mean * norm + state
#             return mean, norm, next_state,estimated_norm
#         else:
#             std = logstd.exp()
#             normal = torch.distributions.Normal(mean,std) 
#             x_t = normal.rsample()
#             next_state = x_t * norm + state
#             return x_t, norm, next_state,estimated_norm
    
    
#     def compute_loss(self,state,action,next_state):
#         info = {}
#         delta_direction = next_state - state
#         delta_direction_norm = torch.norm(delta_direction,dim=-1,keepdim=True)
#         normalized_direction = delta_direction / (delta_direction_norm + 1e-6)

#         mean,logstd, norm,estimated_norm = self.forward(state,action)
#         norm_loss = (norm - delta_direction_norm).pow(2).mean(-1)
#         estimated_norm_loss = (estimated_norm - delta_direction_norm).pow(2).mean(-1)

#         info['norm_loss'] = norm_loss.mean().item()
#         info['estimated_norm_loss'] = estimated_norm_loss.mean().item()

#         if self.deterministic:
#             direction_loss = (mean - normalized_direction).pow(2).mean(-1)
#             loss = direction_loss + norm_loss + estimated_norm_loss
#             info['direction_loss'] = direction_loss.mean().item()
            
#         else:
#             std = logstd.exp()
#             normal = torch.distributions.Normal(mean,std) 
#             log_prob = normal.log_prob(normalized_direction)
#             log_prob = log_prob.sum(-1)   
#             loss = -log_prob + norm_loss
#             info['direction_loss'] = -log_prob.mean().item()

#         return loss,info


# class ForwardModel(nn.Module):
#     def __init__(self,
#         state_dim,
#         action_dim,
#         next_state_dim,
#         hidden_dim = 256,
#         deterministic = False,
#         action_range = None): ### action range is not used in this implementation 
#         super().__init__()
#         self.deterministic = deterministic
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.next_state_dim = next_state_dim
#         init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
#         self.action_lin = Linear(in_features = action_dim, out_features = hidden_dim, **init)
#         self.state_lin = Linear(in_features = state_dim, out_features = hidden_dim, **init)
#         self.delta_direction_layers = nn.Sequential(
#             Linear(in_features = hidden_dim*2, out_features = hidden_dim*2, **init),
#             nn.SiLU(),
#         )
#         self.estimated_norm_layers = nn.Sequential(
#             Linear(in_features = hidden_dim, out_features = hidden_dim*2, **init),
#             nn.SiLU(),  
#         )
#         self.norm_layers = nn.Sequential(
#             Linear(in_features = hidden_dim * 2, out_features = hidden_dim*2, **init),
#             nn.SiLU(),  
#         )
#         if deterministic:
#             self.delta_direction_output_lin = Linear(in_features = hidden_dim*2, out_features = next_state_dim, **init)
#         else:
#             self.delta_direction_outpu_mean = Linear(in_features = hidden_dim*2, out_features = next_state_dim, **init)
#             self.delta_direction_outpu_logstd = Linear(in_features = hidden_dim*2, out_features = next_state_dim, **init)
        
#         self.estimated_norm_output_lin = nn.Sequential(
#                 Linear(in_features = hidden_dim*2, out_features = 1, **init),
#                 nn.ReLU())
#         self.norm_output_lin = nn.Sequential(
#                 Linear(in_features = hidden_dim*2, out_features = 1, **init),
#                 nn.ReLU())

#     def forward(self,state,action):
#         h_s = self.state_lin(state)
#         h_a = self.action_lin(action)
#         h_direction = torch.cat([h_s,h_a],dim=-1)
#         h_direction = self.delta_direction_layers(h_direction) 
#         h_norm = self.norm_layers(h_direction)
        
#         h_estimated_norm = self.estimated_norm_layers(h_s)

#         if self.deterministic:
#             return self.delta_direction_output_lin(h_direction), None ,self.norm_output_lin(h_norm), self.estimated_norm_output_lin(h_estimated_norm)
#         else:
#             return self.delta_direction_outpu_mean(h_direction),self.delta_direction_outpu_logstd(h_direction), self.norm_output_lin(h_norm), self.estimated_norm_output_lin(h_estimated_norm)
        
#     def predict(self,state,action,deterministic = True):
#         mean,logstd, norm, estimated_norm = self.forward(state,action)
#         if deterministic:
#             next_state = mean * norm + state
#             return mean, norm, next_state,estimated_norm
#         else:
#             std = logstd.exp()
#             normal = torch.distributions.Normal(mean,std) 
#             x_t = normal.rsample()
#             next_state = x_t * norm + state
#             return x_t, norm, next_state,estimated_norm
    
    
#     def compute_loss(self,state,action,next_state):
#         info = {}
#         delta_direction = next_state - state
#         delta_direction_norm = torch.norm(delta_direction,dim=-1,keepdim=True)
#         normalized_direction = delta_direction / (delta_direction_norm + 1e-6)

#         mean,logstd, norm,estimated_norm = self.forward(state,action)
#         norm_loss = (norm - delta_direction_norm).pow(2).mean(-1)
#         estimated_norm_loss = (estimated_norm - delta_direction_norm).pow(2).mean(-1)

#         info['norm_loss'] = norm_loss.mean().item()
#         info['estimated_norm_loss'] = estimated_norm_loss.mean().item()

#         if self.deterministic:
#             direction_loss = (mean - normalized_direction).pow(2).mean(-1)
#             loss = direction_loss + norm_loss + estimated_norm_loss
#             info['direction_loss'] = direction_loss.mean().item()
            
#         else:
#             std = logstd.exp()
#             normal = torch.distributions.Normal(mean,std) 
#             log_prob = normal.log_prob(normalized_direction)
#             log_prob = log_prob.sum(-1)   
#             loss = -log_prob + norm_loss
#             info['direction_loss'] = -log_prob.mean().item()

#         return loss,info



# class JointConditionModel(nn.Module):
#     def __init__(self,
#         input_dim,                        
#         condition1_dim   = 0,            # Number of class labels, 0 = unconditional.
#         condition2_dim   = 0,
#         hidden_dim      = 256,              # latent dimension for noise and class labels embedding
#         share_condition_encoder = True
#     ):
#         super().__init__()
#         assert condition1_dim > 0 and condition2_dim > 0, "condition dim should be greater than 0"
#         self.input_dim = input_dim 
#         self.condition1_dim = condition1_dim
#         self.condition2_dim = condition2_dim
#         self.hidden_dim = hidden_dim

#         init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))

#         # Mapping 
#         emb_dim = hidden_dim
#         self.t_step_encoder = nn.Sequential(
#             # PositionalEmbedding(num_channels=emb_dim,max_positions=1000),
#             SinusoidalPosEmb(dim=emb_dim),
#             Linear(in_features=emb_dim, out_features=emb_dim *2, **init),
#             nn.SiLU(),
#             Linear(in_features=emb_dim * 2, out_features=emb_dim , **init))
        
#         condition_emb_dim = emb_dim//2
#         if share_condition_encoder and (condition1_dim == condition2_dim):
#             self.condition1_encoder = nn.Sequential(
#                 Linear(in_features=condition1_dim, out_features=emb_dim, **init),
#                 nn.SiLU(),
#                 Linear(in_features=emb_dim, out_features=condition_emb_dim, **init))
#             self.condition2_encoder = self.condition1_encoder## I'm not sure if this is correct
#         else:
#             self.condition1_encoder = nn.Sequential(
#                 Linear(in_features=condition1_dim, out_features=emb_dim , **init),
#                 nn.SiLU(),
#                 Linear(in_features=emb_dim , out_features=condition_emb_dim, **init))
#             self.condition2_encoder = nn.Sequential(
#                 Linear(in_features=condition2_dim, out_features=emb_dim , **init),
#                 nn.SiLU(),
#                 Linear(in_features=emb_dim, out_features=condition_emb_dim, **init))
#             print("######################## condition1_encoder and condition2_encoder are different ########################")
#         self.condition_emb_dim = condition_emb_dim
#         # Model 
#         self.lin1 = Linear(in_features = input_dim, out_features = hidden_dim, **init)  
#         self.layers = nn.ModuleList([
#             Linear(in_features = hidden_dim, out_features = hidden_dim, **init),
#             Linear(in_features = hidden_dim, out_features = hidden_dim, **init),
#         ])
#         self.output = Linear(in_features = hidden_dim, out_features = input_dim, **init)

#     def forward(self, x,t, condition1 = None,condition2 = None):

#         emb = self.t_step_encoder(t)
#         if condition1 is not None:
#             condition1_emb = self.condition1_encoder(condition1)
#         else:
#             condition1_emb = torch.zeros_like(emb)[:,:self.condition_emb_dim]
        
#         if condition2 is not None:
#             condition2_emb = self.condition2_encoder(condition2)
#         else:
#             condition2_emb = torch.zeros_like(emb)[:,:self.condition_emb_dim]
#         condition_emb = torch.cat([condition1_emb,condition2_emb],dim=-1)
#         emb = emb + condition_emb

#         # encoder 
#         x = F.silu(self.lin1(x) + emb)
#         for layer in self.layers:
#             x = F.silu(layer(x) + emb)
#         x = self.output(x)
#         return x