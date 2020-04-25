#!/usr/bin/env python
# coding: utf-8

# In[3]:


# else dan elif

if <expr>:
    <statement>
else:
    <statement>


# In[2]:


x = 2

if x < 50:
    print('x is small')
else:
    print('x is large')


# In[4]:


x = 20

if x > 50:
    print('x is small')
else:
    print('x is large')


# In[5]:


# One line if Statements

if <expr>:
    <statement>
    
if <expr>: <statement>

if <expr>: <statement1>; <statement2>; <statement3>; <statementn>


# In[6]:


if 'f' in 'foo' : print('1'); print('2'); print('3')


# In[8]:


if a > b
    m = a
else:
    m = b
    
m = a if a > b else b


# In[9]:


n=5

while n>0:
    n-=1
    print(n)


# In[10]:


d = {'foo' : 1, 'bar' : 2, 'baz' :3}

for k in d:
    print(k)


# In[11]:


d['foo']


# In[12]:


for k in d:
    print(d[k])


# In[13]:


d['foo']


# In[14]:


for v in d.values():
    print(v)


# In[17]:


x = range (10)
x


# In[18]:


for n in x:
    print(n)


# In[20]:


print(x)


# In[ ]:


# break dan continue


# In[21]:


for i in ['foo', 'bar', 'baz', 'qux']:
    if 'z' in i:
        break
    print(i)
else:
    print('done')


# In[22]:


temp = input("ketikan temperatur yang ingin dikonversi: ")


# In[23]:


temp


# In[24]:


temp = input("ketikan temperatur yang ingin dikonversi: ")
degree = int(temp)

result = (9*degree)/5+32


# In[25]:


result


# In[26]:


temp = input("ketikan temperatur yang ingin dikonversi, eg.45F,28C: ")
degree = int(temp[:-1])
i_convertion = temp[-1]

if i_convertion == 'C':
    result = (9*degree)/5+32
elif i_convertion == 'F':
    result = (degree-32)*5/9
else:
    print('masukan input yang benar')
    
print("temperaturenya adalah", result)


# In[ ]:


while True:
    msg = input("Ketikan karakter: ").lower()
    print(msg)
    if msg == 'stop' :
        break


# In[ ]:




