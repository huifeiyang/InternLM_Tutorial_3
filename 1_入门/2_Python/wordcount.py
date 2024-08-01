import string


def wordcount(s):

    #缩写复原
    s_no_abbreviation = s.replace("'s"," is")

    # 去除标点符号
    translator = str.maketrans('', '', string.punctuation)
    s_no_punctuation = s_no_abbreviation.translate(translator)

    # 字符串转换成小写
    s_lowercase = s_no_punctuation.lower()

    # 字符串分割成单词列表
    words = s_lowercase.split()

    word_count = {}

    # 遍历和计数
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    return word_count

def main():
    inputword=input("请录入英文短句，不区分大小写:")
    count=wordcount(inputword)
    print (count)


if __name__ == '__main__':
    main()