#!/usr/bin/env python
# coding: utf-8

# print('Hacktiv8')

# In[2]:


print('Hacktiv8')


# In[3]:


## Integer
Integer adalah tipe data di Python yang berupa angka.


# ## Integer

# ## Integer
# 
# Integer adalah tipe data di Python yang berupa angka

# In[6]:


print(123123)


# In[7]:


print(10)


# In[8]:


type(10)


# ## Floating-Point
# 
# Floating point tipe data di Python yang berupa desimal.

# In[9]:


4.2


# In[10]:


type(4.2)


# In[11]:


4.


# In[12]:


.2


# In[13]:


.4e7


# In[14]:


4.2e-4


# ## Strings
# 
# Strings adalah tipe data di python yang berupa sequence of character data.

# In[15]:


print('Ini adalah tipe data string')


# In[16]:


type(Ini adalah tipe data string)


# In[17]:


type('Ini adalah tipe data string')


# In[18]:


print("This string contains a single qoute (') character.")


# ## Boolean
# 
# Boolean adalah tipe data di python yang berupa True or False

# In[19]:


True


# In[20]:


False


# In[21]:


type(True)


# In[22]:


type(False)


# In[23]:


type('True')


# In[24]:


type(1)


# In[25]:


type('1')


# ## Variables

# In[27]:


n = 300


# In[28]:


n


# In[29]:


print(n)


# In[30]:


n = k = 300
print(n, k)


# ### Variables Names

# In[33]:


name = 'Raka'
Age = 26
has_laptops = True
print(name, Age, has_laptops)


# In[34]:


bola_naga_9 = True


# In[35]:


age = 1
Age = 2
aGe = 3
AGE = 4
a_g_e = 5
_age = 6
age_ = 7
_AGE_ = 8

print(age, Age, aGe, AGE, a_g_e, _age, age_, _AGE_)


# ## Operators and Expressions in Python

# In[ ]:





# In[36]:


a = 10
b = 20


# In[37]:


a + b


# a + b - 5

# In[40]:


a = 4
b = 3

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** b)


# In[41]:


10/5


# ### Comparison Operators

# In[43]:


a=10
b=20

print(a==b)
print(a!=b)
print(a<=b)
print(a>=b)


# ### String Manipulaion

# In[45]:


#+operators
s='foo'
t='bar'
u='baz'

print(s+t)


# In[46]:


s+t+u


# In[48]:


print('hactiv8 '+'Inalum')


# In[49]:


#+operators
s*4


# In[51]:


#inoperators
s='foo'
s in 'That food for us'


# In[52]:


s in 'That good for us'


# In[55]:


# Case Conversion

s = 'HacKTIvAte iNAlum'


# In[56]:


#Capitalize

s.capitalize()


# In[57]:


#Title

s.title()


# In[58]:


#lower
print(s.lower())

#UPPERCASE
print(s.upper())


# In[60]:


#Swapcase
s.swapcase()


# ## List
# 

# In[61]:


a = ['foo' , 'bar', 'baz', 'qux']


# In[62]:


a


# In[63]:


a = [21.42, 'foobar', 3, 4, 'bark', False, 3.14159]


# In[64]:


a


# In[65]:


b = ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']


# In[66]:


b


# In[67]:


b[0]


# In[68]:


b[5]


# In[69]:


b[-1]


# In[70]:


b[-6]


# In[71]:


#slicing

b[2:5]


# In[73]:


b[:5]


# In[74]:


b[2:]


# In[75]:


print(b)


# In[76]:


b + ['valeu2', 'value2']


# In[77]:


b*2


# In[78]:


print(b)

len(b)


# In[79]:


min(b)


# In[80]:


max(b)


# ### Modifying single list value

# In[82]:


b


# In[85]:


b[2] = 10


# In[86]:


b


# In[87]:


b[-1]=20


# In[88]:


b


# In[89]:


del b[3]


# In[90]:


b


# ### Modifying Multiple List Values

# In[91]:


b


# In[92]:


b[1:4]


# In[93]:


b[1:4] = [1.1, 2.2, 3.3]


# In[94]:


b


# ## Tuples

# In[95]:


t = ('foo', 'bar', 'baz', 'qux', 'quux', 'corge')
t


# In[98]:


t[0]


# In[99]:


t[-1]


# ### Dictionary

# In[104]:


identitas = {
    'Fnama' : 'Widi',
    'Lname': 'Cahyati',
    'Age' : 27
}


# In[105]:


identitas


# In[106]:


identitas['Age']


# In[109]:


identitas['Fnama']


# In[110]:


identitas['Pekerjaan'] = 'Guru'


# In[112]:


identitas['Pekerjaan']


# In[115]:


identitas['Pekerjaan'] = 'Data Scientist'


# In[116]:


identitas


# In[117]:


del identitas['Pekerjaan']


# In[118]:


identitas


# In[119]:


person = {}


# In[120]:


type(person)


# In[121]:


person['fname']='Nama Depan'
person['lname']='Nama Belakang'
person['age']=40
person['children']=['Anak 1', 'Anak 2', 'Anak 3']


# In[122]:


person


# In[123]:


person['pets'] = {'dog' :'dogName', 'Cat' : 'CatName'}


# In[124]:


person


# In[125]:


person['age']


# In[126]:


person['children']


# In[127]:


person['children'][1]


# In[128]:


person['pets']


# In[130]:


person['pets']['Cat']


# In[133]:


d = [1, 2, 3 [4, 5, 6], 7]


# In[134]:





# In[136]:


# Python Statement

print('Hello World')

x = [1, 2, 3]


# In[137]:


# Line Continuation

person1_age = 42
person2_age = 16
person3_age = 71


# In[140]:


someone_is_of_working_age = (
    (person1_age >= 18 and person1_age <= 65) or
    (person2_age >= 18 and person2_age <= 65) or
    (person3_age >= 18 and person3_age <= 65)
)

someone_is_of_working_age


# In[141]:


a = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]


# In[142]:


a


# In[144]:


s ='Hello World'


# In[145]:


s


# In[147]:


# Multiple Statement per Line

x=1;y=2;z=3

print(x); print(y); print(z)


# In[150]:


# Comments

a=['foo', 'bar'] # variable a berisi list foo dan bar


# In[149]:


a


# In[151]:


b = '# im not a comment'


# In[152]:


b


# In[153]:


"""
Initiative value for radius of circle

Then calculate the area of the circle
and display the result to the notebook
"""

pi = 3.14
r = 12

area = pi * (r**2)
print('The area of circle is', area)


# In[ ]:




