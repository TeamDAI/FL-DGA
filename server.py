from glob_inc.server_fl import *
import json
import paho.mqtt.client as client
from model_api.src.FashionMnist import aggregated_models

def on_connect(client, userdata, flags, rc):
    print_log("Connected with result code "+str(rc))

def on_disconnect(client, userdata, rc):
    print_log("Disconnected with result code "+str(rc))
    # Kết nối lại khi mất kết nối
    client.reconnect()

def on_message(client, userdata, msg):
    print(f"received msg from {msg.topic}")
    topic = msg.topic
    if topic == "dynamicFL/join":
        handle_join(client, userdata, msg)
    elif "dynamicFL/res" in topic:
        tmp = topic.split("/")
        this_client_id = tmp[2]
        handle_res(this_client_id, msg)

def handle_res(this_client_id, msg):
    data = json.loads(msg.payload)
    cmd = data["task"]
    if cmd == "EVA_CONN":
        print_log(f"{this_client_id} complete task EVA_CONN")
        handle_pingres(this_client_id, msg)
    elif cmd == "TRAIN":
        print_log(f"{this_client_id} complete task TRAIN")
        handle_trainres(this_client_id, msg)
    elif cmd == "WRITE_MODEL":
        print_log(f"{this_client_id} complete task WRITE_MODEL")
        handle_update_writemodel(this_client_id, msg)

def handle_join(client, userdata, msg):
    this_client_id = msg.payload.decode("utf-8")
    print("joined from"+" "+this_client_id)
    client_dict[this_client_id] = {
        "state": "joined"
    }
    server.subscribe(topic="dynamicFL/res/"+this_client_id)
    

def handle_pingres(this_client_id, msg):
    print(msg.topic+" "+str(msg.payload.decode()))
    ping_res = json.loads(msg.payload)
    this_client_id = ping_res["client_id"]
    if ping_res["packet_loss"] == 0.0:
        print_log(f"{this_client_id} is a good client")
        state = client_dict[this_client_id]["state"]
        print_log(f"state {this_client_id}: {state}, round: {n_round}")
        if state == "joined" or state == "trained":
            client_dict[this_client_id]["state"] = "eva_conn_ok"
            #send_model("saved_model/FashionMnist.pt", server, this_client_id)
            #print(client_dict)
            count_eva_conn_ok = sum(1 for client_info in client_dict.values() if client_info["state"] == "eva_conn_ok")
            if(count_eva_conn_ok == NUM_DEVICE):
                print_log("publish to " + "dynamicFL/model/all_client")
                send_model("saved_model/LSTMModel.pt", server, this_client_id)
   
def handle_trainres(this_client_id, msg):
    payload = json.loads(msg.payload.decode())
    
    client_trainres_dict[this_client_id] = payload["weight"]
    state = client_dict[this_client_id]["state"]
    if state == "model_recv":
        client_dict[this_client_id]["state"] = "trained"

    #print("done train!")
    
def handle_update_writemodel(this_client_id, msg):
    state = client_dict[this_client_id]["state"]
    if state == "eva_conn_ok":
        client_dict[this_client_id]["state"] = "model_recv"
        send_task("TRAIN", server, this_client_id)
        count_model_recv = sum(1 for client_info in client_dict.values() if client_info["state"] == "model_recv")
        if(count_model_recv == NUM_DEVICE):
            print_log("Waiting for training from client...")
        



def start_round():
    global n_round
    n_round = n_round + 1
    print_log(f"server start round {n_round}")
    round_state = "started"
    for client_i in client_dict:
        send_task("EVA_CONN", server, client_i)

    #t = threading.Timer(round_duration, end_round)
    #t.start()
    #wait rev all model from client before end round
    while (len(client_trainres_dict) != NUM_DEVICE):
        time.sleep(1)
    time.sleep(5)
    end_round()

    

def do_aggregate():
    print_log("Do aggregate ...")
    aggregated_models(client_trainres_dict,n_round)
    
def handle_next_round_duration():
    #if len(client_trainres_dict) < len(client_dict):
        #time_between_two_round = time_between_two_round + 10
    while (len(client_trainres_dict) < NUM_DEVICE):
        time.sleep(1)

def end_round():
    global n_round
    print_log(f"server end round {n_round}")
    round_state = "finished"
    if n_round < NUM_ROUND:
        handle_next_round_duration()
        do_aggregate()
        t = threading.Timer(time_between_two_round, start_round)
        t.start()
    else:
        do_aggregate()
        for c in client_dict:
            send_task("STOP", server, c)
            print_log(f"send task STOP {c}")
        server.loop_stop()

def on_subscribe(mosq, obj, mid, granted_qos):
    print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

if __name__ == "__main__":
   
    NUM_ROUND = 3
    NUM_DEVICE = 2
    global global_model
    client_dict = {}
    client_trainres_dict = {}
    round_duration = 50
    time_between_two_round = 30
    round_state = "finished"

    server = client.Client(client_id="server")
    server.connect(broker_name, port=1883, keepalive=300)

    server.on_connect = on_connect
    server.on_disconnect = on_disconnect
    server.on_message = on_message
    server.on_subscribe = on_subscribe
 
    server.loop_start()
    server.subscribe(topic="dynamicFL/join")
    print_log(f"server sub to dynamicFL/join of {broker_name}")

    print_log("server is waiting for clients to join the topic ...")

    while (NUM_DEVICE > len(client_dict)):
       time.sleep(1)

    start_round()
    server._thread.join()
    time.sleep(10)
    print_log("server exits")
