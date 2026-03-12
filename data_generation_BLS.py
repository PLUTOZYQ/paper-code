from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, pair
import time
import numpy as np
import csv
import os

# 初始化群
group = PairingGroup('SS512')

def bls_keygen():
    """BLS密钥生成"""
    g1 = group.random(G1)
    g2 = group.random(G2)
    sk = group.random(ZR)
    pk = g2 ** sk
    return (sk, pk, g1, g2)

def bls_sign(sk, m):
    """BLS签名"""
    h = group.hash(m, G1)
    sig = h ** sk
    return sig

def bls_verify(pk, m, sig, g1, g2):
    """BLS验证"""
    h = group.hash(m, G1)
    return pair(sig, g2) == pair(h, pk)

def bls_aggregate(sigs):
    """聚合签名"""
    agg_sig = group.init(G1)
    for sig in sigs:
        agg_sig *= sig
    return agg_sig

def bls_aggregate_verify(pks, msgs, agg_sig, g1, g2):
    """聚合验证"""
    h = group.init(G1)
    for msg in msgs:
        h *= group.hash(msg, G1)
    agg_pk = group.init(G2)
    for pk in pks:
        agg_pk *= pk
    return pair(agg_sig, g2) == pair(h, agg_pk)

def test_bls_aggregation(participant_counts, message_length=1024, iterations=100):
    """测试BLS聚合性能"""
    agg_sign_times = []
    agg_verify_times = []
    single_verify_times = []
    
    # 获取单签名验证时间作为基线
    single_verify_time = []
    single_sign_time = []  # 新增：记录单个签名时间
    for _ in range(iterations):
        sk, pk, g1, g2 = bls_keygen()
        msg = 'a' * message_length
        
        # 测试单个签名时间
        start_time = time.perf_counter()
        sig = bls_sign(sk, msg)
        single_sign_time.append(time.perf_counter() - start_time)
        
        # 测试单个验证时间
        start_time = time.perf_counter()
        bls_verify(pk, msg, sig, g1, g2)
        single_verify_time.append(time.perf_counter() - start_time)
    
    single_verify_base = np.mean(single_verify_time)
    single_sign_base = np.mean(single_sign_time)  # 单个签名平均时间

    for count in participant_counts:
        agg_sign_time = []
        agg_verify_time = []

        for _ in range(iterations):
            # 生成密钥对
            keys = [bls_keygen() for _ in range(count)]
            sks = [key[0] for key in keys]
            pks = [key[1] for key in keys]
            g1, g2 = keys[0][2], keys[0][3]

            # 生成消息和单个签名
            msgs = ['a' * message_length for _ in range(count)]
            
            # 生成所有单个签名（用于聚合）
            sigs = []
            for sk, msg in zip(sks, msgs):
                sigs.append(bls_sign(sk, msg))

            # 测试聚合签名生成时间
            start_time = time.perf_counter()
            agg_sig = bls_aggregate(sigs)
            agg_sign_time.append(time.perf_counter() - start_time)

            # 测试聚合签名验证时间
            start_time = time.perf_counter()
            valid = bls_aggregate_verify(pks, msgs, agg_sig, g1, g2)
            agg_verify_time.append(time.perf_counter() - start_time)

        # 计算平均时间
        agg_sign_mean = np.mean(agg_sign_time)
        agg_verify_mean = np.mean(agg_verify_time)
        
        # 计算单签名验证总时间（n个独立验证）
        single_verify_total = single_verify_base * count
        # 计算单签名生成总时间（n个独立签名）
        single_sign_total = single_sign_base * count
        
        agg_sign_times.append(agg_sign_mean)
        agg_verify_times.append(agg_verify_mean)
        single_verify_times.append(single_verify_total)
        
        print(f"测试完成 {count} 个参与者: 聚合签名时间={agg_sign_mean:.6f}s, 聚合验证时间={agg_verify_mean:.6f}s, 顺序验证总时间={single_verify_total:.6f}s")

    return agg_sign_times, agg_verify_times, single_verify_times, single_sign_base, single_verify_base

def save_to_csv(participant_counts, agg_sign_times, agg_verify_times, single_verify_times, 
                single_sign_base, single_verify_base, message_length=1024):
    """将测试结果保存到CSV文件"""
    
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 计算签名长度数据
    signature_length = 64  # BLS签名在SS512群中的长度（字节）
    single_sig_length = signature_length
    agg_sig_length = signature_length  # BLS聚合签名长度固定
    
    # 计算压缩比率
    compression_ratios = []
    for count in participant_counts:
        total_single_length = single_sig_length * count
        compression_ratio = agg_sig_length / total_single_length if total_single_length > 0 else 0
        compression_ratios.append(compression_ratio)
    
    # 计算验证效率比
    verify_efficiency_ratios = []
    for i, count in enumerate(participant_counts):
        if agg_verify_times[i] > 0:
            ratio = single_verify_times[i] / agg_verify_times[i]
        else:
            ratio = 0
        verify_efficiency_ratios.append(ratio)
    
    # 计算签名效率比
    sign_efficiency_ratios = []
    for i, count in enumerate(participant_counts):
        single_sign_total = single_sign_base * count
        if agg_sign_times[i] > 0:
            ratio = single_sign_total / agg_sign_times[i]
        else:
            ratio = 0
        sign_efficiency_ratios.append(ratio)
    
    # 计算时间比（签名/验证）
    sign_verify_time_ratios = []
    for i in range(len(participant_counts)):
        if agg_verify_times[i] > 0:
            ratio = agg_sign_times[i] / agg_verify_times[i]
        else:
            ratio = 0
        sign_verify_time_ratios.append(ratio)
    
    # 1. 保存签名长度数据
    with open('data/bls_signature_length.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['participants', 'single_sig_length', 'agg_sig_length', 'compression_ratio'])
        for i, count in enumerate(participant_counts):
            writer.writerow([count, single_sig_length, agg_sig_length, compression_ratios[i]])
    print(f"签名长度数据已保存到 data/bls_signature_length.csv")
    
    # 2. 保存签名时间数据
    with open('data/bls_signing_time.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['participants', 'sequential_sign_mean', 'aggregate_sign_mean', 'sign_efficiency_ratio'])
        for i, count in enumerate(participant_counts):
            sequential_sign = single_sign_base * count
            writer.writerow([count, sequential_sign, agg_sign_times[i], sign_efficiency_ratios[i]])
    print(f"签名时间数据已保存到 data/bls_signing_time.csv")
    
    # 3. 保存验证时间数据
    with open('data/bls_verification_time.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['participants', 'sequential_verify_mean', 'aggregate_verify_mean', 
                         'verify_efficiency_ratio', 'sign_verify_time_ratio'])
        for i, count in enumerate(participant_counts):
            writer.writerow([count, single_verify_times[i], agg_verify_times[i], 
                            verify_efficiency_ratios[i], sign_verify_time_ratios[i]])
    print(f"验证时间数据已保存到 data/bls_verification_time.csv")
    
    return {
        'signature_length': signature_length,
        'single_sign_base': single_sign_base,
        'single_verify_base': single_verify_base,
        'compression_ratios': compression_ratios,
        'sign_efficiency_ratios': sign_efficiency_ratios,
        'verify_efficiency_ratios': verify_efficiency_ratios,
        'sign_verify_time_ratios': sign_verify_time_ratios
    }

def generate_test_data():
    """生成测试数据的主函数"""
    # 增加数据点，使曲线更饱满
    # 增加更大范围的参与者数量，以展示小规模和大规模参与者的情况
    participant_counts = list(range(1, 101, 3))  # 每3个参与者一个点
    participant_counts.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 增加小规模关键点
    participant_counts.extend([15, 25, 35, 45, 55, 65, 75, 85, 95])  # 增加中等规模关键点
    participant_counts.extend([120, 150, 180, 200])  # 增加大规模关键点
    participant_counts = sorted(list(set(participant_counts)))  # 去重并排序
    
    print("=" * 60)
    print("开始生成BLS聚合签名测试数据...")
    print(f"测试参与者数量: {participant_counts}")
    print(f"参与者数量范围: {min(participant_counts)} - {max(participant_counts)}")
    print(f"消息长度: 1024 字节")
    print(f"每个参与者数量测试迭代次数: 100 次")
    print("=" * 60)
    
    # 进行测试
    agg_sign_times, agg_verify_times, single_verify_times, single_sign_base, single_verify_base = test_bls_aggregation(participant_counts)
    
    # 保存数据到CSV文件
    results = save_to_csv(participant_counts, agg_sign_times, agg_verify_times, single_verify_times,
                         single_sign_base, single_verify_base)
    
    # 打印性能摘要
    print("\n性能摘要:")
    print("=" * 100)
    print(f"{'参与者数量':<12} {'聚合签名(s)':<15} {'顺序签名(s)':<15} {'聚合验证(s)':<15} {'顺序验证(s)':<15} {'签名效率比':<15} {'验证效率比':<15}")
    print("-" * 100)
    
    # 打印关键点
    key_points = [1, 5, 10, 25, 50, 75, 100, 150, 200]
    for count in key_points:
        if count in participant_counts:
            idx = participant_counts.index(count)
            sequential_sign = single_sign_base * count
            print(f"{count:<12} {agg_sign_times[idx]:<15.6f} {sequential_sign:<15.6f} "
                  f"{agg_verify_times[idx]:<15.6f} {single_verify_times[idx]:<15.6f} "
                  f"{results['sign_efficiency_ratios'][idx]:<15.2f} {results['verify_efficiency_ratios'][idx]:<15.2f}")
    
    print("=" * 100)
    print(f"单个签名时间: {single_sign_base:.6f}s")
    print(f"单个验证时间: {single_verify_base:.6f}s")
    print(f"BLS签名长度: {results['signature_length']} 字节")
    print(f"最大聚合签名时间: {max(agg_sign_times):.6f}s (参与者数量={participant_counts[agg_sign_times.index(max(agg_sign_times))]})")
    print(f"最大聚合验证时间: {max(agg_verify_times):.6f}s (参与者数量={participant_counts[agg_verify_times.index(max(agg_verify_times))]})")
    print(f"最高签名效率比: {max(results['sign_efficiency_ratios']):.2f} (参与者数量={participant_counts[results['sign_efficiency_ratios'].index(max(results['sign_efficiency_ratios']))]})")
    print(f"最高验证效率比: {max(results['verify_efficiency_ratios']):.2f} (参与者数量={participant_counts[results['verify_efficiency_ratios'].index(max(results['verify_efficiency_ratios']))]})")
    print(f"数据已保存到 'data/' 目录")
    
    return participant_counts, agg_sign_times, agg_verify_times, single_verify_times, results

# 主程序
if __name__ == "__main__":
    generate_test_data()