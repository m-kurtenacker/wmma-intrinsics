from IPython import embed

import sys

import pandas as pd
#pd.set_option("display.max_rows", 10000)

assert(len(sys.argv) >= 3) # <name> input output or <name> input filter output


df = pd.read_csv(sys.argv[1], sep=r",|_", header=None, engine="python")

# Format: test/bench_bxy_2_2_wxy_1_1_labc_row_col_row_ks_1_sk_0_alchemist_tiled,110.908997
#df = df.drop(columns=[0, 1, 4, 7, 11, 13]).rename(columns={2: "x", 3: "y", 5: "wx", 6: "wy", 8: "la", 9: "lb", 10: "lc", 12: "ks", 14: "sk", 15: "target", 16: "name", 17: "time"}).assign(pb=1)
# Format: test/bench_bxy_2_2_wxy_1_1_pb_4_labc_row_col_row_ks_1_sk_0_alchemist_tiled,110.908997
df = df.drop(columns=[0, 1, 4, 7, 9, 13, 15]).rename(columns={2: "x", 3: "y", 5: "wx", 6: "wy", 8: "pb", 10: "la", 11: "lb", 12: "lc", 14: "ks", 16: "sk", 17: "target", 18: "name", 19: "time"})

#Go through the failed tests list and remove invalid entries!
if len(sys.argv) > 3:
    filter_file = sys.argv[2]
    filter_df = pd.read_csv(filter_file, sep=r"_", header=None)
    #filter_df = filter_df.drop(columns=[0, 1, 4, 7, 11, 13, 17]).rename(columns={2: "x", 3: "y", 5: "wx", 6: "wy", 8: "la", 9: "lb", 10: "lc", 12: "ks", 14: "sk", 15: "target", 16: "name"}).assign(pb=1)
    filter_df = filter_df.drop(columns=[0, 1, 4, 7, 9, 13, 15, 19]).rename(columns={2: "x", 3: "y", 5: "wx", 6: "wy", 8: "pb", 10: "la", 11: "lb", 12: "lc", 14: "ks", 16: "sk", 17: "target", 18: "name"})
    df = pd.merge(df, filter_df, how="outer", indicator=True).query('_merge == "left_only"').drop(columns=["_merge"])


if len(sys.argv) > 3:
    output_file_name = sys.argv[3]
else:
    output_file_name = sys.argv[2]

try:
    df2 = pd.read_feather(output_file_name)
    df = pd.concat([df2, df])
    print(df)
    df.to_feather(output_file_name)
except FileNotFoundError:
    df.to_feather(output_file_name)
