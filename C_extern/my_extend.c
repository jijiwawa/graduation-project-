#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<Python.h>

int my_abs(int n){
    if(n<0)
        n = n * -1;
    return n;
}

void my_reverse(char *s){
    if(s){
        int len = strlen(s);
        int i;
        char t;
        for(i= 0;i<(len-1)/2;i++){
            t = s[i];
            s[i] = s[len-1-i];
            s[len-1-i] = t;
        }
    }

}

void test(void){
    printf("test my_abs:\n");
    printf("|-8590|=%d\n",my_abs(-8590));
    printf("|-0|=%d\n",my_abs(-0));
    printf("|5690|=%d\n",my_abs(-5690));

    printf("test my_reverse:\n");
    char s0[10] = "apple";
    char s1[20] = "I love you!";
    char *s2 = NULL;
    my_reverse(s0);
    my_reverse(s1);
    my_reverse(s2);
    printf("'apple' reverse is '%s'\n",s0);
    printf("'I love you!' reverse is '%s'\n",s1);
    printf("null reverse is %s\n",s2);
}

//作用，接受python传的值，将结果计算后转为Python对象返回给python
//返回类型PyObject*,函数名：模块名_函数名
static PyObject *Extest_abs(PyObject *self,PyObject *args){
    int num;
    if(!(PyArg_ParseTuple(args,"i",&num))){
        return NULL;
    }
    return (PyObject*)Py_BuildValue("i",my_abs(num));
}

static PyObject *Extest_reverse(PyObject *self,PyObject *args){
    char *s;
    if(!(PyArg_ParseTuple(args,"z",&s))){
        return NULL;
    }
    my_reverse(s);
    return (PyObject*)Py_BuildValue("s",s);
}

static PyObject *Extest_test(PyObject *self,PyObject *args){
    test();
    return (PyObject*)Py_BuildValue("");
}

//为每个模块增加PyMethodDef ModuleMethods[]数组
static PyMethodDef ExtestMethods[] = {
    {"abs",Extest_abs,METH_VARARGS},
    {"reverse",Extest_reverse,METH_VARARGS},
    {"test",Extest_test,METH_VARARGS},
    {NULL,NULL},
};

static struct PyModuleDef ExtestModule = {
    PyModuleDef_HEAD_INIT,
    "Extest",
    NULL,
    -1,
    ExtestMethods
};

void PyInit_Extest(){
    PyModule_Create(&ExtestModule);
}