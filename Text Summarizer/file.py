##with open('test.txt','r') as f:
##    f_contents = f.read()
##    print(f_contents,end='')
##
##    for line in f:
##        print(line,end ='')
    
with open('test.txt','r') as rf:
    with open('output.txt','w') as wf:
        chunk_Size = 10;
        r_content = rf.read(chunk_Size)
        while len(r_content) > 0:
            wf.write(r_content)
            r_content = rf.read(chunk_Size)
