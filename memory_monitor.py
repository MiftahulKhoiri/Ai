import os
import time
import threading

def monitor_memory(interval=2):
    """Monitor penggunaan memori di background"""
    def _monitor():
        while True:
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if 'VmRSS' in line:
                            mem_kb = int(line.split()[1])
                            mem_mb = mem_kb / 1024
                            print(f"[MEMORY] {mem_mb:.1f} MB")
                            if mem_mb > 500:  # Warning jika > 500MB
                                print("[WARNING] Memori tinggi, pertimbangkan untuk mengurangi batch size!")
            except:
                pass
            time.sleep(interval)
    
    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()