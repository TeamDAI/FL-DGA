from glob_inc.server_fl import *
import json
import paho.mqtt.client as client
#from model_api.src.ml_api import aggregated_models

def on_connect(client, userdata, flags, rc):
    print_log("Connected with result code "+str(rc))

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
    # this_client_id = this_client_id.decode("utf-8")
    data = json.loads(msg.payload)
    cmd = data["task"]
    if cmd == "EVA_CONN":
        handle_pingres(this_client_id, msg)
    elif cmd == "TRAIN":
        handle_trainres(this_client_id, msg)
    elif cmd == "WRITE_MODEL":
        handle_update_writemodel(this_client_id, msg)

def handle_join(client, userdata, msg):
    this_client_id = msg.payload.decode("utf-8")
    print("joined from"+" "+this_client_id)
    client_dict[this_client_id] = {
        "state": "joined"
    }
    server.subscribe(topic="dynamicFL/res/"+this_client_id)
    # print(client_dict)
    

def handle_pingres(this_client_id, msg):
    print(msg.topic+" "+str(msg.payload.decode()))
    # ping_res = msg.payload.decode("utf-8")
    ping_res = json.loads(msg.payload)
    this_client_id = ping_res["client_id"]
    # print(ping_res["clientid"])
    if ping_res["packet_loss"] == 0.0:
        print_log(f"{this_client_id} is a good client")
        state = client_dict[this_client_id]["state"]
        print(state)
        if state == "joined" or state == "trained":
            client_dict[this_client_id]["state"] = "eva_conn_ok"
            send_model("saved_model/FashionMnist.pt", server, this_client_id)
        # time.sleep(10)
        # send_task("TRAIN", client)
        # start_time = threading.Timer(5, send_task, args=["EVA_CONN", client])
        # print_log("server wait for newround")
        # start_time.start()
    # print(client_dict)

def handle_trainres(this_client_id, msg):
    #print("trainres"+" "+str(msg.payload))
    print("Trainres")
    payload = json.loads(msg.payload.decode())
    
    client_trainres_dict[this_client_id] = payload["weight"]
    state = client_dict[this_client_id]["state"]
    if state == "model_recv":
        client_dict[this_client_id]["state"] = "trained"

    print("done train!")
    #print(client_trainres_dict)
    # print(client_dict)
    
def handle_update_writemodel(this_client_id, msg):
    state = client_dict[this_client_id]["state"]
    if state == "eva_conn_ok":
        client_dict[this_client_id]["state"] = "model_recv"
        send_task("TRAIN", server, this_client_id)
    # print(client_dict)

def start_round():
    global n_round
    n_round = n_round + 1
    print_log(f"server start round {n_round}")
    round_state = "started"
    for client_i in client_dict:
        send_task("EVA_CONN", server, client_i)
    t = threading.Timer(round_duration, end_round)
    t.start()

def do_aggregate():
    #aggregated_models(client_trainres_dict, )
    print("do_aggregate")

def handle_next_round_duration():
    if len(client_trainres_dict) < len(client_dict):
        time_between_two_round = time_between_two_round + 10

def end_round():
    global n_round
    print_log(f"server end round {n_round}")
    round_state = "finished"
    if n_round <= NUM_ROUND:
        handle_next_round_duration()
        do_aggregate()
        t = threading.Timer(time_between_two_round, start_round)
        t.start()
    else:
        for c in client_dict:
            send_task("STOP", server, c)

def on_subscribe(mosq, obj, mid, granted_qos):
    print_log("Subscribed: " + str(mid) + " " + str(granted_qos))

if __name__ == "__main__":
   
    NUM_ROUND=2
    client_dict = {}
    client_trainres_dict = {}
    round_duration = 30
    time_between_two_round = 30
    round_state = "finished"

    server = client.Client(client_id="server")
    server.connect(broker_name)

    server.on_connect = on_connect
    server.on_message = on_message
    server.on_subscribe = on_subscribe
    # server.message_callback_add("dynamicFL/join", handle_join)
    # server.message_callback_add("dynamicFL/ping_res", handle_pingres)
    # server.message_callback_add("dynamicFL/train_res", handle_trainres)

    #server.loop_start()
    server.loop_start()
    server.subscribe(topic="dynamicFL/join")
    print_log(f"server sub to dynamicFL/join of {broker_name}")

    print("WAITTING ...")

    #server._thread.join()

    # while n_round < 11:
    #while True:
       #time.sleep(10)

    time.sleep(10)
    start_round()
    # end_round_timer = threading.Timer(round_duration, end_round)

    # while n_round < 11:
    #     pass
    # print_log("server exits")
