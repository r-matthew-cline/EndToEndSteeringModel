import os

os.system("python steering_model.py --model CNN[5533]_drop --action train")
os.system("python steering_model.py --model CNN[5533]_nodrop --action train")
os.system("python steering_model.py --model CNN[3333]_drop --action train")
os.system("python steering_model.py --model CNN[3333]_nodrop --action train")
