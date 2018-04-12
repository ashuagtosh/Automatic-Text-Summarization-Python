import os

def readList(file_name):
    global script_dir
    script_dir = os.path.dirname(__file__)
    rel_path = "documents/"+file_name
    abs_file_path = os.path.join(script_dir, rel_path)
    rf = open(abs_file_path,"r")
    L = list()
    L = rf.readlines()
    L = list(map(lambda s: s.strip(), L))
    return L
def createMatrix(list_name):
    length = len(list_name)
    w = 5
    h = length
    matrix = [[0 for x in range(w)] for y in range(h)]
    k = 0
    for i in range(0,h):
        matrix[k][0] = list_name[k][0]
        matrix[k][1] = list_name[k][1]
        k = k + 1
    #print(matrix)
    return
def main():
    global list1
    list1 = list()
    global list2
    list2 = list()
    global list3
    list3 = list()
    global list4
    list4 = list()
    global list5
    list5 = list()
    global list6
    list6 = list()
    
    list1= readList("tokenize_document1.txt")
    list2= readList("tokenize_document2.txt")
    list3= readList("tokenize_document3.txt")
    list4= readList("tokenize_document4.txt")

    createMatrix(list1)

    return

main()
