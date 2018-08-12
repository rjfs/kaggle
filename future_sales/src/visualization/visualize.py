import pandas as pd
import seaborn as sns


def series_bars(series):
    vc = [s.value_counts() for s in series]
    vcs = pd.concat(vc, axis=1).unstack()
    vcs.name = 'count'
    plt_df = vcs.reset_index()
    plt_df = plt_df.rename(columns={'level_0': 'set', 'level_1': 'category'})
    # Plot
    g = sns.catplot(
        x='category', y="count", hue="set", data=plt_df,
        height=6, kind="bar", palette="muted"
    )
    g.fig.set_size_inches(16, 8)
    g.despine(left=True)


def compare_dists(train_df, test_df, train_month=None):
    if train_month is None:
        train_month = train_df['date_block_num'].max()
    train_m = train_df[train_df['date_block_num'] == train_month]
    s1 = train_m['shop_id']
    s1.name = 'train_month'
    s2 = test_df['shop_id']
    s2.name = 'test'
    series_bars([s1, s2])

