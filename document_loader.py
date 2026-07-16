import json
import csv

class DocumentLoader:
    @staticmethod
    def _load_text(file_path):
        '''读取文件内容，返回字符串，文件不存在则返回None'''
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"错误：文件‘{file_path}'不存在")
            return None
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                    print(f"错误：文件‘{file_path}'无法以utf-8或者gbk解码，错误信息：{e}")
                    return None
        except Exception as e:
                print(f"错误：读取文件‘{file_path}'时发生异常，错误信息：{e}")
                return None

    @staticmethod
    def _load_json(file_path):        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误：文件‘{file_path}'不存在")
            return None
        except json.JSONDecodeError as e:
            print(f"错误：文件‘{file_path}'不是有效的JSON格式. 错误信息：{e}")
            return None
        except Exception as e:
            print(f"错误：读取文件‘{file_path}'时发生异常，错误信息：{e}")
            return None
        
    @staticmethod
    def _load_csv(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                return list(reader)
        except FileNotFoundError:
            print(f"错误：文件‘{file_path}'不存在")
            return None
        except csv.Error as e:
            print(f"错误：文件‘{file_path}'不是有效的CSV格式. 错误信息：{e}")
            return None
        except Exception as e:
            print(f"错误：读取文件‘{file_path}'时发生异常，错误信息：{e}")
            return None
        
    @staticmethod
    def load(file_path):
        '''根据文件扩展名选择加载方式，返回内容，文件不存在或格式错误则返回None'''
        if file_path.endswith('.json'):
            return DocumentLoader._load_json(file_path)
        elif file_path.endswith('.txt'):
            return DocumentLoader._load_text(file_path)
        elif file_path.endswith('.csv'):
            return DocumentLoader._load_csv(file_path)
        else:
            print(f"错误：不支持的文件类型‘{file_path}'")
            return None