from glob_inc.utils import *
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import paho.mqtt.client
import time
import threading

broker_name = "100.82.9.118"
#broker_name = "192.168.10.128"
n_round = 0

def send_task(task_name, client, this_client_id):
    print_log("publish to " + "dynamicFL/req/"+this_client_id)
    client.publish(topic="dynamicFL/req/"+this_client_id, payload=task_name)
def send_model(path, client, this_client_id):
    f = open(path, "rb")
    data = f.read()
    f.close()
    #print_log(f"sent model to {this_client_id} with len = {len(data)}b")
    #print_log("publish to " + "dynamicFL/model/"+this_client_id)
    #print_log("publish to " + "dynamicFL/model/all_client")
    #client.publish(topic="dynamicFL/model/"+this_client_id, payload=data)    
    client.publish(topic="dynamicFL/model/all_client", payload=data)





