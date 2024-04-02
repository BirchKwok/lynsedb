from pyroaring import BitMap


class IDFilter:
    def __init__(self):
        self.ids = BitMap()

    def add(self, items):
        if isinstance(items, int):  # 如果items是单个整数
            self.ids.add(items)
        else:  # 如果items是可迭代的整数集合
            self.ids.update(items)

    def drop(self, item):
        self.ids.discard(item)

    def __contains__(self, item):
        return item in self.ids

    def to_file(self, filepath):
        # 使用serialize方法序列化BitMap为bytes，然后写入文件
        with open(filepath, 'wb') as file:
            file.write(self.ids.serialize())

    def from_file(self, filepath):
        try:
            with open(filepath, 'rb') as file:
                # 从文件读取bytes，然后使用deserialize方法反序列化为BitMap对象
                self.ids = BitMap.deserialize(file.read())
        except FileNotFoundError:
            self.ids = BitMap()

    def find_max_value(self):
        return max(self.ids) if self.ids else -1
