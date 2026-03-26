import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_maniskill_eval(data_dir):
    data = pd.read_csv(data_dir)
    actor_type = data['actor_type'].unique().tolist()
    if 'noised' in actor_type:
        para = 'actor_noise_scale'
    elif 'laggy' in actor_type:
        para = 'actor_repeat_prob'
    else:
        para = 'actor_eps'
    actor_type = actor_type[0]
    print(type(actor_type))

    data_grouped = data.groupby(["expname", "noise_level","actor_type", para]).agg(
    success_rate_mean=("success_rate", "mean")).reset_index() 

    def create_method_name(row):
        if row["expname"] == "User":
            return "User"
        else:
            return f"CM ({row['noise_level']}/80)"

    data_grouped["method"] = data_grouped.apply(create_method_name, axis=1)
    filtered_data = data_grouped[data_grouped["actor_type"] == actor_type]

    sns.set(style="whitegrid")

    # Create a bar plot for success rate comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=filtered_data,
        x=para,
        y="success_rate_mean",
        hue="method",
        # palette="viridis",
    )
    # Annotate each bar with its height
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

    # Customize the plot
    plt.title("Success Rate Comparison Across Methods")
    plt.xlabel(f"Env Parameter: {para}")
    plt.ylabel("Average Success Rate")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()


