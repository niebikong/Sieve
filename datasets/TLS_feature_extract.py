from flowcontainer.extractor import extract
import os
import csv
import numpy as np
import multiprocessing
from multiprocessing import Pool

def get_time_diff(x):
    if [round(x[i+1]-x[i],4) for i in range(len(x)-1)] == []:
        return [0]
    else:
        return [round(x[i+1]-x[i],4) for i in range(len(x)-1)]
    
def get_fwd_time_list(list_timestamps,list_length):
    a = np.where(np.array(list_length)>0)
    b = list(np.array(list_timestamps)[a])
    return(get_time_diff(b))

def get_backward_time_list(list_timestamps, list_length):
    a = np.where(np.array(list_length)<0)
    b = list(np.array(list_timestamps)[a])
    return(get_time_diff(b))

def get_direction_time(timestamps, lengths, direction):
    if direction == 'fwd':
        mask = lengths > 0
    else:
        mask = lengths < 0
    filtered = timestamps[mask]
    return get_time_diff(filtered)

def safe_stat(func, arr, default=0):
    return round(func(arr).item(), 4) if len(arr) > 0 else default



def get_1D_vector(file_path, label):
    result = extract(file_path)  
    vectors = []
    
    for key in result:
        value = result[key]
        ###众数计算
        IP_fwd_packet_length_Most = [0]
        if list(filter(lambda x: x>0, value.ip_lengths)) != []:
            IP_fwd_packet_length_Most = np.argmax(np.bincount(list(filter(lambda x: x>0, value.ip_lengths))))

        IP_backward_packet_length_Most = [0]
        if list(filter(lambda x: x>0, [-x for x in value.ip_lengths])) != []:
            IP_backward_packet_length_Most = np.argmax(np.bincount(list(filter(lambda x: x>0, [-x for x in value.ip_lengths]))))

        payload_fwd_packet_length_Most = [0]
        if list(filter(lambda x: x>0, value.payload_lengths)) != []:
            payload_fwd_packet_length_Most = np.argmax(np.bincount(list(filter(lambda x: x>0, value.payload_lengths))))

        payload_backward_packet_length_Most = [0]
        if list(filter(lambda x: x>0, [-x for x in value.payload_lengths])) != []:
            payload_backward_packet_length_Most = np.argmax(np.bincount(list(filter(lambda x: x>0, [-x for x in value.payload_lengths]))))

        IP_flow_duration = 10000000
        if value.ip_timestamps[-1] - value.ip_timestamps[0] != 0:
            IP_flow_duration = value.ip_timestamps[-1] - value.ip_timestamps[0]
        IP_flow_duration = max(IP_flow_duration, 1e-6)  # 避免除以零

        payload_flow_duration = 10000000
        if value.payload_timestamps[-1] - value.payload_timestamps[0] != 0:
            payload_flow_duration = value.payload_timestamps[-1] - value.payload_timestamps[0]
        payload_flow_duration = max(payload_flow_duration, 1e-6)

        # 预处理IP相关数据
        ip_lengths = np.array(value.ip_lengths, dtype=np.float64)
        ip_lengths = np.nan_to_num(ip_lengths, nan=0)
        ip_timestamps = np.array(value.ip_timestamps)
        ip_timestamps.sort()
        
        ip_lengths_fwd = ip_lengths[ip_lengths > 0]
        ip_lengths_backward = -ip_lengths[ip_lengths < 0]
        
        # 预处理Payload相关数据
        payload_lengths = np.array(value.payload_lengths, dtype=np.float64)
        payload_lengths = np.nan_to_num(payload_lengths, nan=0)
        payload_timestamps = np.array(value.payload_timestamps)
        payload_timestamps.sort()
        
        payload_lengths_fwd = payload_lengths[payload_lengths > 0]
        payload_lengths_backward = -payload_lengths[payload_lengths < 0]
        

        
        IP_time_diff_all = get_time_diff(value.ip_timestamps)  ## IP包双向到达时间间隔
        IP_time_diff_fwd = get_fwd_time_list(value.ip_timestamps,value.ip_lengths) ##IP包前向到达时间间隔
        IP_time_diff_backward = get_backward_time_list(value.ip_timestamps,value.ip_lengths)##IP包后向到达时间间隔

        payload_time_diff_all = get_time_diff(value.payload_timestamps) ## payload包双向到达时间间隔
        payload_time_diff_fwd = get_fwd_time_list(value.payload_timestamps,value.payload_lengths) ##payload包前向到达时间间隔
        payload_time_diff_backward = get_backward_time_list(value.payload_timestamps,value.payload_lengths)##payload包后向到达时间间隔

        
        # 流持续时间
        ip_duration = max(ip_timestamps[-1] - ip_timestamps[0], 1e-6) if len(ip_timestamps) > 1 else 1e-6
        payload_duration = max(payload_timestamps[-1] - payload_timestamps[0], 1e-6) if len(payload_timestamps) > 1 else 1e-6
        
        payload_flow_duration = max(payload_flow_duration, 1e-6)

        # 构造特征向量
        vector = [
            # 时间相关特征
            round(ip_duration, 4),
            round(payload_duration, 4),
            
            # IP长度统计
            len(ip_lengths_fwd),
            len(ip_lengths_backward),
            len(ip_lengths),
            np.sum(ip_lengths_fwd),
            np.sum(ip_lengths_backward),
            np.sum(np.abs(ip_lengths)),
            safe_stat(np.max, ip_lengths_fwd),
            safe_stat(np.min, ip_lengths_fwd, 1500),
            safe_stat(np.mean, ip_lengths_fwd),
            safe_stat(np.std, ip_lengths_fwd),
            safe_stat(np.var, ip_lengths_fwd),
            safe_stat(np.median, ip_lengths_fwd),
            
            # 后向包统计（省略部分特征，根据原代码补充完整）
            # 类似地处理其他统计量...
            IP_fwd_packet_length_Most,
            -min(value.ip_lengths),
            -max(i if i < 0 else -1500 for i in value.ip_lengths),
            -round(np.mean(list(filter(lambda x: x < 0, value.ip_lengths))), 4),
            round(np.std(list(filter(lambda x: x < 0, value.ip_lengths))), 4),
            round(np.var(list(filter(lambda x: x < 0, value.ip_lengths))), 4),
            -np.median(list(filter(lambda x: x < 0, value.ip_lengths))),
            IP_backward_packet_length_Most,
            round(len(value.ip_lengths) / IP_flow_duration, 4),
            round(sum([i > 0 for i in value.ip_lengths]) / IP_flow_duration, 4),
            round(sum([i < 0 for i in value.ip_lengths]) / IP_flow_duration, 4),
            round(sum(i if i > 0 else -i for i in value.ip_lengths) / IP_flow_duration, 4),
            round(sum(i if i > 0 else 0 for i in value.ip_lengths) / IP_flow_duration, 4),
            round(-sum(i if i < 0 else 0 for i in value.ip_lengths) / IP_flow_duration, 4),
            max(IP_time_diff_fwd),
            min(IP_time_diff_fwd),
            round(np.mean(IP_time_diff_fwd), 4),
            round(np.std(IP_time_diff_fwd), 4),
            round(np.var(IP_time_diff_fwd), 4),
            round(np.median(IP_time_diff_fwd), 4),
            max(IP_time_diff_backward),
            min(IP_time_diff_backward),
            round(np.mean(IP_time_diff_backward), 4),
            round(np.std(IP_time_diff_backward), 4),
            round(np.var(IP_time_diff_backward), 4),
            round(np.median(IP_time_diff_backward), 4),
            round((value.payload_timestamps[-1] - value.payload_timestamps[0]), 4),
            sum([i > 0 for i in value.payload_lengths]),
            sum([i < 0 for i in value.payload_lengths]),
            len(value.payload_lengths),
            sum(i if i > 0 else 0 for i in value.payload_lengths),
            -sum(i if i < 0 else 0 for i in value.payload_lengths),
            sum(i if i > 0 else -i for i in value.payload_lengths),
            max(value.payload_lengths),
            min(i if i > 0 else 1500 for i in value.payload_lengths),
            round(np.mean(list(filter(lambda x: x > 0, value.payload_lengths))), 4),
            round(np.std(list(filter(lambda x: x > 0, value.payload_lengths))), 4),
            round(np.var(list(filter(lambda x: x > 0, value.payload_lengths))), 4),
            np.median(list(filter(lambda x: x > 0, value.payload_lengths))),
            payload_fwd_packet_length_Most,
            -min(value.payload_lengths),
            -max(i if i < 0 else -1500 for i in value.payload_lengths),
            -round(np.mean(list(filter(lambda x: x < 0, value.payload_lengths))), 4),
            round(np.std(list(filter(lambda x: x < 0, value.payload_lengths))), 4),
            round(np.var(list(filter(lambda x: x < 0, value.payload_lengths))), 4),
            -np.median(list(filter(lambda x: x < 0, value.payload_lengths))),
            payload_backward_packet_length_Most,
            # 流量速率
            round(len(ip_lengths) / ip_duration, 4),

            round(sum([i > 0 for i in value.payload_lengths]) / payload_flow_duration, 4),
            round(sum([i < 0 for i in value.payload_lengths]) / payload_flow_duration, 4),
            round(sum(i if i > 0 else -i for i in value.payload_lengths) / payload_flow_duration, 4),
            round(sum(i if i > 0 else 0 for i in value.payload_lengths) / payload_flow_duration, 4),
            round(-sum(i if i < 0 else 0 for i in value.payload_lengths) / payload_flow_duration, 4),
            max(payload_time_diff_fwd),
            min(payload_time_diff_fwd),
            round(np.mean(payload_time_diff_fwd), 4),
            round(np.std(payload_time_diff_fwd), 4),
            round(np.var(payload_time_diff_fwd), 4),
            round(np.median(payload_time_diff_fwd), 4),
            max(payload_time_diff_backward),
            min(payload_time_diff_backward),
            round(np.mean(payload_time_diff_backward), 4),
            round(np.std(payload_time_diff_backward), 4),
            round(np.var(payload_time_diff_backward), 4),
            round(np.median(payload_time_diff_backward), 4),

            label
        ]
        vectors.append(vector)
    
    return vectors


def process_file(args):
    pcap_path, label = args
    try:
        vectors = get_1D_vector(pcap_path, label)
        return vectors
    except Exception as e:
        print(f"Error processing {pcap_path}: {str(e)}")
        return []

def write_vectors_to_csv(root_path, output_csv):
    traffic_names = os.listdir(root_path)
    traffic_dict = {name: idx for idx, name in enumerate(traffic_names)}

    # 打开输出 CSV 文件
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header_columns = [str(i) for i in range(1, 80)]
        header_columns.append('label')

        # 写入表头（根据实际特征顺序补充）
        writer.writerow(header_columns)

        # 创建一个进程池来并行处理每个文件
        pool = Pool(processes=multiprocessing.cpu_count())  # 使用所有可用的CPU核心
        
        tasks = []
        for category in traffic_names:
            label = traffic_dict[category]
            category_path = os.path.join(root_path, category)
            
            for pcap_file in os.listdir(category_path):
                pcap_path = os.path.join(category_path, pcap_file)
                tasks.append((pcap_path, label))  # 构建任务列表
        
        # 使用进程池并行处理文件
        all_vectors = pool.map(process_file, tasks)
        
        # 将处理结果写入 CSV 文件
        for vectors in all_vectors:
            for vector in vectors:
                writer.writerow(vector)
        
        pool.close()
        pool.join()  # 等待所有进程完成

if __name__ == '__main__':
    root_path = r'D:\迅雷下载\恶意流量数据集\malicious_TLS_4_paper'
    output_csv = r'D:\python项目\flowcontainer-master (1)\flowcontainer-master\特征提取\CSV\MAL_TLS2023.csv'
    write_vectors_to_csv(root_path, output_csv)