# -*- coding: utf-8 -*-

"""
Created on Wed Jun  1 10:27:03 2016

@author: anmeng

"""
import random
import lmdb

env_out = lmdb.open("/home/jkj/caffe/caffe-ssd/examples/lmdb/trainval_lmdb", map_size=int(1e12)) 
env1 = lmdb.open("/home/jkj/caffe/caffe-ssd/examples/coco/coco_train_lmdb",readonly=True)
env2 = lmdb.open("/home/jkj/caffe/caffe-ssd/examples/coco/coco_val_lmdb",readonly=True)
env3 = lmdb.open("/home/jkj/caffe/caffe-ssd/examples/VOC0712/VOC0712_trainval_lmdb",readonly=True)


#env.set_mapsize(1000L*1000L*1000L*16*10) #扩大映射范围，才可以追加





tempdata=[]
count=int(0)
for env in env1,env2,env3:
	print(env.stat()) #状态

	txn = env.begin()
	database = txn.cursor()
	#env.open_db(key="newDBName", txn=txn)
	#newDatabase = txt.cursor("newDBName")
	for (key, value) in database:
	    tempdata.append((key,value))
	    count+=1
	

print count
random.shuffle(tempdata)
print len(tempdata) 
count=0
with env_out.begin(write=True) as txn_out:
	for (key, value) in tempdata:
		print key
		count+=1
		print count
		txn_out.put(key,value)
		    
    
env_out.close()
env.close()
env2.close()

 
print("success")
