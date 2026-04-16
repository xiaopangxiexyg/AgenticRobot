## H1-2 control

The files in this folder should be put in H1-2 AGX to achieve the communication between AGX and H1's PC1.

file structure

```bash
.
├── arm_controller.py     # control the arm of the H1 robot
├── hand_controller.py    # control the hand of the H1 robot (inspire hand)
├── server.py               # socket server to get the data from teleop data
└── utils.py              # some defined data and functions
```

First, you should open the H1 inspire service in the H1's PC2.


**Please install the [unitree_dds_wrapper](https://github.com/unitreerobotics/unitree_dds_wrapper)** in the H1's AGX.


You should run the `server.py` in the H1-2 AGX to start the server. The server will listen to the port 12345 and wait for the data from the teleop.
Please make sure the ip address binded in the `server.py` is the ip address of the H1-2 AGX. You can change the ip by giving the ip address as the argument when running the `server.py`.

example:
```bash
python server.py --ip 127.0.0.1
```
After the server is started, you can run the script in the teleop to send the data to the H1-2 AGX.

The data format is as follows:

```json
{
    "arm": [
        "left arm qpos 7d vector",
        "right arm qpos 7d vector",
    ],
    "hand": [
        "left hand qpos 6d vector",
        "right hand qpos 6d vector",
    ]
}
```

The `arm_controller.py` and `hand_controller.py` will parse the data and send the data to the H1 robot.
