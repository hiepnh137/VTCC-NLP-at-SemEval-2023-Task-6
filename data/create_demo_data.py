import json

train_data = json.load(open('train.json'))
train_demo_data = train_data[:3]
json.dump(train_demo_data, open('train_demo.json', 'w'), indent=4)

dev_data = json.load(open('dev.json'))
dev_demo_data = dev_data[:3]
json.dump(dev_demo_data, open('dev_demo.json', 'w'), indent=4)