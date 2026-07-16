import argparse
from collections import Counter
import re

def read_file(file_path):
    '''读取文件内容，返回字符串，文件不存在则返回None'''
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误：文件‘{file_path}'不存在")
        return None
    

def count_words(text):
    '''统计文本中单词出现的次数，返回Counter对象'''
    #使用正则表达式把文本拆分成单词列表（只保留字母和数字）
    words = re.findall(r'\b\w+\b', text.lower())
    return Counter(words)

def main():
    parser = argparse.ArgumentParser(description='统计文本文件中单词出现的次数')
    parser.add_argument('file_path', help='要统计的文本文件路径')
    parser.add_argument("-n", "--n", type=int, default=10, help='显示出现次数最多的前N个单词，默认为10')

    #解析参数
    args = parser.parse_args()

    #读取文件内容
    text = read_file(args.file_path) 
    if text is None:
        return #文件读取出错直接结束
    
    #统计词频
    counts = count_words(text)

    #打印出现频率最高的N个词
    print(f"出现频率最高的{args.n}个单词：")
    for word, count in counts.most_common(args.n):
        print(f"{word}: {count}")

if __name__ == '__main__':
    main()

