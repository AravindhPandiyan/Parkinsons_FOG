# This is to add the subnet routing to the gateway router on the routing table in the container.
ip route add 10.0.0.0/24 via 10.0.1.2

# This is the code to be run after the above.
python api_main.py