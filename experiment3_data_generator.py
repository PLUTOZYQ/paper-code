"""
Experiment 3: Data Generator for Different Number of Messages
"""
import random
import numpy as np
import json
from Crypto.Util.number import getPrime, inverse, GCD
import gmpy2
from functools import reduce
import time
import os


# =============== Paillier Encryption System ===============
def lcm(a, b):
    return abs(a * b) // GCD(a, b)

def L(x, n):
    return (x - 1) // n

class Paillier:
    def __init__(self, bit_length):
        self.p = getPrime(bit_length // 2)
        self.q = getPrime(bit_length // 2)
        self.n = self.p * self.q
        self.nsqr = self.n * self.n
        self.lmbda = lcm(self.p - 1, self.q - 1)
        self.g = self.n + 1
        x = pow(self.g, self.lmbda, self.nsqr)
        self.mu = inverse(L(x, self.n), self.n)
    
    def encrypt(self, m):
        r = random.randint(1, self.n)
        c = pow(self.g, m, self.nsqr) * pow(r, self.n, self.nsqr) % self.nsqr
        return c


# =============== Aggregation Schemes ===============
def traditional_aggregate(c_list, paillier):
    result = 1
    for c in c_list:
        result *= c
        result = result % paillier.nsqr
    return result

def reduce_aggregate(c_list, paillier):
    result = reduce(lambda x, y: (x * y) % paillier.nsqr, c_list)
    return result

def optimized_aggregation(c_list, paillier):
    a = len(c_list)
    b = gmpy2.mpz(1)
    matrix = np.array(c_list, dtype=object)
    if a < 4:
        result = reduce(lambda x, y: gmpy2.mul(x, y) % paillier.nsqr, matrix)
    else:
        if a % 2 == 1:
            b = matrix[-1]
            matrix = np.delete(matrix, -1)
        mid = a // 2
        left_matrix = matrix[:mid]
        right_matrix = matrix[mid:]
        result_matrix = [gmpy2.mul(l, r) % paillier.nsqr for l, r in zip(left_matrix, right_matrix)]
        result = reduce(lambda x, y: gmpy2.mul(x, y) % paillier.nsqr, result_matrix) * b % paillier.nsqr
    return result


# =============== Data Generation Function ===============
def generate_random_large_integers(l, w):
    return [random.getrandbits(w) for _ in range(l)]


# =============== Remove Outliers Function ===============
def remove_outliers(data):
    data = np.array(data)
    if len(data) == 0:
        return 0
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered_data = [x for x in data if (mean - 2 * std_dev < x < mean + 2 * std_dev)]
    return np.mean(filtered_data) if filtered_data else mean


# =============== Generate Experiment Data ===============
def generate_experiment_data():
    """Generate experiment data for different number of messages"""
    print("=" * 60)
    print("Generating Data for Experiment 3: Different Number of Messages")
    print("=" * 60)
    
    # Experiment parameters
    key_length = 1024
    message_bits = 8 * 1024  # 8KB
    trials = 10
    
    # Message count list (logarithmic distribution)
    message_count_list = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    
    # Store results
    results = {
        'experiment_name': 'Message_Count_Comparison',
        'parameters': {
            'key_length': key_length,
            'message_bits': message_bits,
            'trials': trials,
            'message_counts': message_count_list
        },
        'traditional_times': [],
        'optimized_times': [],
        'reduce_times': [],
        'raw_data': {}
    }
    
    # Run experiment
    for message_count in message_count_list:
        print(f"Testing with {message_count} messages")
        
        traditional_trial_times = []
        optimized_trial_times = []
        reduce_trial_times = []
        
        for trial in range(trials):
            # Initialize Paillier system
            paillier = Paillier(key_length)
            
            # Generate random plaintexts
            numbers = generate_random_large_integers(message_count, message_bits)
            
            # Encrypt all plaintexts
            encrypted_numbers = [paillier.encrypt(number) for number in numbers]
            
            # Test traditional scheme
            start = time.perf_counter()
            traditional_aggregate(encrypted_numbers, paillier)
            traditional_trial_times.append((time.perf_counter() - start) * 1000)
            
            # Test optimized scheme
            start = time.perf_counter()
            optimized_aggregation(encrypted_numbers, paillier)
            optimized_trial_times.append((time.perf_counter() - start) * 1000)
            
            # Test reduce scheme
            start = time.perf_counter()
            reduce_aggregate(encrypted_numbers, paillier)
            reduce_trial_times.append((time.perf_counter() - start) * 1000)
        
        # Remove outliers and record average time
        traditional_avg = remove_outliers(traditional_trial_times)
        optimized_avg = remove_outliers(optimized_trial_times)
        reduce_avg = remove_outliers(reduce_trial_times)
        
        results['traditional_times'].append(float(traditional_avg))
        results['optimized_times'].append(float(optimized_avg))
        results['reduce_times'].append(float(reduce_avg))
        
        # Store raw data for reference
        results['raw_data'][str(message_count)] = {
            'traditional_raw': [float(t) for t in traditional_trial_times],
            'optimized_raw': [float(t) for t in optimized_trial_times],
            'reduce_raw': [float(t) for t in reduce_trial_times]
        }
        
        print(f"  Traditional: {traditional_avg:.4f} ms")
        print(f"  Optimized: {optimized_avg:.4f} ms")
        print(f"  Reduce: {reduce_avg:.4f} ms")
    
    # Create data directory if it doesn't exist
    os.makedirs('experiment_data', exist_ok=True)
    
    # Save data to JSON file
    data_file = 'experiment_data/exp3_message_count_data.json'
    with open(data_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nData saved to: {data_file}")
    return results


# =============== Main Program ===============
if __name__ == "__main__":
    print("Experiment 3 Data Generator")
    print("-" * 60)
    print("Parameters:")
    print(f"  Key Length: 1024 bits")
    print(f"  Message Size: {8*1024} bits (8KB)")
    print(f"  Trials per Test: 10")
    print(f"  Message Counts: 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000")
    print("-" * 60)
    
    # Generate data
    results = generate_experiment_data()
    
    print("\n" + "=" * 60)
    print("Data generation completed!")
    print("=" * 60)