from glob_inc.client_fl import *
import paho.mqtt.client as client
import sys

def on_connect(client, userdata, flags, rc):
    print_log("Connected with result code "+str(rc))

def on_message(client, userdata, msg):
    print_log(f"on_message {client._client_id.decode()}")
    print_log("RECEIVED msg from " + msg.topic)
    topic = msg.topic
    if topic == "dynamicFL/req/"+client_id:
        handle_cmd(client, userdata, msg)
    # elif topic == "dynamicFL/model/"+client_id:
    #     handle_model(client, userdata, msg)

def on_subscribe(client, userdata, mid, granted_qos):
    print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

if __name__ == "__main__":

    client_id = "client_" + sys.argv[1]
    print(client_id)
    fl_client = client.Client(client_id=client_id)
    fl_client.connect(broker_name)
    fl_client.on_connect = on_connect
    fl_client.on_message = on_message
    fl_client.on_subscribe = on_subscribe

    # fl_client.message_callback_add("dynamicFL/cmd", handle_cmd)
    fl_client.message_callback_add("dynamicFL/model/"+client_id, handle_model)
    
    fl_client.loop_start()
    # fl_client.subscribe(topic="dynamicFL/join")
    fl_client.subscribe(topic="dynamicFL/model/"+client_id)
    fl_client.subscribe(topic="dynamicFL/req/"+client_id)
    # join_dFL_topic(fl_client)
    fl_client.publish(topic="dynamicFL/join", payload=client_id)
    print_log(f"{client_id} joined dynamicFL/join of {broker_name}")

    # print_log("main start sleeping 100s")
    # time.sleep(100)
    fl_client._thread.join()

    #fl_client.loop_stop()
    print_log("client exits")