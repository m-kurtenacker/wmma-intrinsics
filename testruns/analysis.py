from IPython import embed

import sys

import pandas as pd
pd.set_option("display.max_rows", 10000)

assert(len(sys.argv) > 1)


df = pd.read_feather(sys.argv[1])

print("Input data size:", df.shape)



# Compare tiled and shmem results. Row layout no alchemist and col layout on battlemage seem to yield best results for classical tiling w/o shared mem.
#query1 = df.query('name == "tiled" and ((target == "alchemist" and lb == "row") or (target == "battlemage" and lb == "col"))').assign(ks=-1).groupby(by=["target", "name", "ks", "x", "y", "wx", "wy", "lb"])
#query2 = df.query('target == "alchemist" and name == "shmem" and sk == 16').groupby(by=["target", "name", "ks", "x", "y", "wx", "wy", "lb"])

#result1 = query1.agg(time=("time", "median")).sort_values("time")
#result2 = query2.agg(time=("time", "median")).sort_values("time")
#result = pd.concat([result1, result2]).sort_values("time")



# Per device, find best layout for tiled & shared
#query1 = df.query('target == "alchemist" and name == "tiled"').groupby(by=["target", "name", "x", "y", "wx", "wy", "lb"])
#query2 = df.query('target == "alchemist" and name == "shmem"').groupby(by=["target", "name", "ks", "sk", "x", "y", "wx", "wy", "lb"])
#query3 = df.query('target == "battlemage" and name == "tiled"').groupby(by=["target", "name", "x", "y", "wx", "wy", "lb"])
#query4 = df.query('target == "battlemage" and name == "shmem"').groupby(by=["target", "name", "ks", "sk", "x", "y", "wx", "wy", "lb"])

#result1 = query1.agg(time=("time", "median")).sort_values("time")
#result2 = query2.agg(time=("time", "median")).sort_values("time")
#result3 = query3.agg(time=("time", "median")).sort_values("time")
#result4 = query4.agg(time=("time", "median")).sort_values("time")



# Per device, find best layout for tiled, shared, blocked
query1 = df.query('target == "alchemist" and name == "tiled"').assign(ks=-1, sk=-1, pb=-1).groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])
query2 = df.query('target == "alchemist" and name == "shmem"').assign(pb=-1).groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])
query3 = df.query('target == "alchemist" and name == "blocked"').groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])
query4 = df.query('target == "alchemist" and name == "wmma"').assign(x=-1, y=-1, wx=-1, wy=-1, ks=-1, sk=-1).groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])

result1 = query1.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)
result2 = query2.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)
result3 = query3.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)
result4 = query4.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)

print("Best individual runs on alchemist")
print(result1.head(15))
print(result2.head(15))
print(result3.head(15))
print(result4.head(15))

result_alchemist = pd.concat([result1, result2, result3, result4]).sort_values("time")


query1 = df.query('target == "battlemage" and name == "tiled"').assign(ks=-1, sk=-1, pb=-1).groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])
query2 = df.query('target == "battlemage" and name == "shmem"').assign(pb=-1).groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])
query3 = df.query('target == "battlemage" and name == "blocked"').groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])
query4 = df.query('target == "battlemage" and name == "wmma"').assign(x=-1, y=-1, wx=-1, wy=-1, ks=-1, sk=-1).groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])

result1 = query1.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)
result2 = query2.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)
result3 = query3.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)
result4 = query4.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)

print("Best individual runs on battlemage")
print(result1.head(15))
print(result2.head(15))
print(result3.head(15))
print(result4.head(15))

result_battlemage = pd.concat([result1, result2, result3, result4]).sort_values("time")


query1 = df.query('target == "rtx-2080" and name == "tiled"').assign(ks=-1, sk=-1, pb=-1).groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])
query2 = df.query('target == "rtx-2080" and name == "blocked"').groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])
query3 = df.query('target == "rtx-2080" and name == "wmma"').assign(x=-1, y=-1, wx=-1, wy=-1, ks=-1, sk=-1).groupby(by=["target", "name", "pb", "ks", "sk", "x", "y", "wx", "wy", "lb"])

result1 = query1.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)
result2 = query2.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)
result3 = query3.agg(time=("time", "median")).sort_values("time").assign(Tflops=lambda df: (4096 * 4096 * 4096 * 2)/(df.time/1000)/1000000000000)

print("Best individual runs on rtx 2080")
print(result1.head(15))
print(result2.head(15))
print(result3.head(15))

result_rtx = pd.concat([result1, result2, result3]).sort_values("time")


# Compare different versions, all of the same layout.
#query1 = df.query('target == "alchemist" and name == "tiled"').assign(sk=0, ks=1).groupby(by=["name", "pb", "x", "y", "wx", "wy", "lb"])
#query2 = df.query('target == "alchemist" and name != "tiled" and sk == 0 and ks == 1').groupby(by=["name", "pb", "x", "y", "wx", "wy", "lb"])

#result1 = query1.agg(time=("time", "median"))
#result2 = query2.agg(time=("time", "median"))
#result = pd.concat([result1, result2]).groupby(by=["x", "y", "wx", "wy", "lb"], group_keys=False)


#To figure out what target/name combinations are available.
#result = df.groupby(by=["target", "name"]).agg(time=("time", "median"))


#for r in result:
#    print(r)

#print(result)
#print(result1.head(15))
#print(result2.head(15))

#print("Best runs overall")
#print(result_alchemist.head(15))
#print(result_battlemage.head(15))
#print(result_rtx.head(15))

#embed()
