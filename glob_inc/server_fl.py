from glob_inc.utils import *
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import paho.mqtt.client
import time
import threading

broker_host = "192.168.10.129"
broker_name = "192.168.10.129"
# broker_name = "192.168.139.129"
n_round = 0
def send_task(task_name, client, this_client_id):
    # publish.single(topic="dynamicFL/cmd", payload=task_name, hostname=broker_name, client_id=server_id)
    print_log("publish to " + "dynamicFL/req/"+this_client_id)
    client.publish(topic="dynamicFL/req/"+this_client_id, payload=task_name)
def send_model(path, client, this_client_id):
    f = open(path, "rb")
    data = f.read()
    f.close()
    print_log(len(data))
    print_log("publish to " + "dynamicFL/model/"+this_client_id)
    client.publish(topic="dynamicFL/model/"+this_client_id, payload=data)




