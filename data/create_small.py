import json
def create_data_subset(data_dir,cats):
  with open(data_dir+"/annotations.json") as f:
    data = json.load(f)
  data_subset_cats=[]
  for x in data["categories"]:
    if x["name"] in cats:
      data_subset_cats.append(x)

  reqd_cats = [x["id"] for x in data_subset_cats]
  print("INFO: Cats are: ",[x["name"] for x in data_subset_cats])
  print("INFO: CatIDs are: ",reqd_cats)
  print("INFO: len of cats:",len(reqd_cats))
  data_subset_anns = []
  reqd_imgs = set()
  for x in data["annotations"]:
    if x["category_id"] in reqd_cats:
      data_subset_anns.append(x)
      reqd_imgs.add(x["image_id"])
  data_subset_imgs = []
  for x in data["images"]:
    if x["id"] in reqd_imgs:
      data_subset_imgs.append(x)
  reqd_dict = {}
  reqd_dict["info"] = {}
  reqd_dict["images"]=data_subset_imgs
  reqd_dict["categories"]=data_subset_cats
  reqd_dict["annotations"]=data_subset_anns

  with open(data_dir+"/annotations-small.json","w") as f:
    json.dump(reqd_dict, f)
create_data_subset("./train",cats=["water","grapes","tomato","banana","carrot","bread-white","butter","coffee-with-caffeine","egg","apple","salad-leaf-salad-green","bread-wholemeal"])
create_data_subset("./val",cats=["water","grapes","tomato","banana","carrot","bread-white","butter","coffee-with-caffeine","egg","apple","salad-leaf-salad-green","bread-wholemeal"])