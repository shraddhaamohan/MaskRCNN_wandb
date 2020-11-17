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
create_data_subset("./train",cats=['potatoes-steamed', 'chips-french-fries', 'mixed-vegetables', 'mixed-salad-chopped-without-sauce', 'leaf-spinach', 'salad-leaf-salad-green', 'avocado', 'french-beans', 'cucumber', 'sweet-pepper', 'tomato', 'zucchini', 'carrot', 'broccoli', 'apple', 'banana', 'strawberries', 'hard-cheese', 'cheese', 'rice', 'pasta-spaghetti', 'bread-whole-wheat', 'bread-wholemeal', 'bread-white', 'chicken', 'egg', 'butter', 'jam', 'dark-chocolate', 'tea', 'espresso-with-caffeine', 'coffee-with-caffeine', 'white-coffee-with-caffeine', 'water', 'water-mineral', 'wine-red', 'wine-white', 'tomato-sauce', 'mayonnaise', 'pizza-margherita-baked'])
create_data_subset("./val",cats=['potatoes-steamed', 'chips-french-fries', 'mixed-vegetables', 'mixed-salad-chopped-without-sauce', 'leaf-spinach', 'salad-leaf-salad-green', 'avocado', 'french-beans', 'cucumber', 'sweet-pepper', 'tomato', 'zucchini', 'carrot', 'broccoli', 'apple', 'banana', 'strawberries', 'hard-cheese', 'cheese', 'rice', 'pasta-spaghetti', 'bread-whole-wheat', 'bread-wholemeal', 'bread-white', 'chicken', 'egg', 'butter', 'jam', 'dark-chocolate', 'tea', 'espresso-with-caffeine', 'coffee-with-caffeine', 'white-coffee-with-caffeine', 'water', 'water-mineral', 'wine-red', 'wine-white', 'tomato-sauce', 'mayonnaise', 'pizza-margherita-baked'])