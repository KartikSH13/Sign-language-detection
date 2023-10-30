import pandas as pd
import os
import heapq

main_path = "videos/"
batch_path = "batch_data/"

wlasl_df = pd.read_json(batch_path + "batch_1.json")

def check(name, video_data, count_dict):
    count = 0
    for row in video_data:
        video_path = f'{main_path}{row["video_id"]}.mp4'
        if os.path.exists(video_path):
            count += 1
    count_dict[name] = count

count_dict = {}

for index, row in wlasl_df.iterrows():
    check(row['gloss'], row['instances'], count_dict)

# Get top 5 and top 10 entries based on counts
top_5_gloss = heapq.nlargest(5, count_dict, key=count_dict.get)
top_10_gloss = heapq.nlargest(10, count_dict, key=count_dict.get)


print("\nTop 10 Gloss Terms:")
for term in top_10_gloss:
    print(f"{term}: {count_dict[term]} videos")

# print("Top 5 Gloss Terms:")
# for term in top_5_gloss:
#     print(f"{term}: {count_dict[term]} videos")