#!/usr/bin/env python3
import math
import sys
from typing import Callable, Tuple, List

SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
SAFE_MATH.update({"abs": abs, "min": min, "max": max})

def parse_function(expr: str) -> Callable[[float], float]:
    def f(x):
        return eval(expr, {"__builtins__": {}}, dict(SAFE_MATH, x=x))
    return f

def numerical_derivative(f, x, h=1e-6):
    return (f(x+h)-f(x-h))/(2*h)

def print_iteration_table(rows):
    print("{:<8} {:<20} {:<20} {:<20}".format("Iter","x","f(x)","Error"))
    print("-"*70)
    for it,x,fx,err in rows:
        print(f"{it:<8d} {x:<20.12g} {fx:<20.12g} {err:<20.12g}")

def bisection(f,a,b,tol,max_iter=100):
    fa,fb=f(a),f(b)
    if fa*fb>0: raise ValueError("The function must have opposite signs at a and b.")
    rows=[]
    for i in range(1,max_iter+1):
        c=(a+b)/2; fc=f(c); err=abs(b-a)/2
        rows.append((i,c,fc,err))
        if abs(fc)<tol or err<tol: return c,err,i,rows
        if fa*fc<0: b=c; fb=fc
        else: a=c; fa=fc
    return c,err,max_iter,rows

def regula_falsi(f,a,b,tol,max_iter=100):
    fa,fb=f(a),f(b)
    if fa*fb>0: raise ValueError("The function must have opposite signs at a and b.")
    rows=[]; c=a
    for i in range(1,max_iter+1):
        c_prev=c
        c=(a*fb-b*fa)/(fb-fa)
        fc=f(c)
        err=abs(c-c_prev) if i>1 else float("inf")
        rows.append((i,c,fc,err))
        if abs(fc)<tol or err<tol: return c,err,i,rows
        if fa*fc<0: b=c; fb=fc
        else: a=c; fa=fc
    return c,err,max_iter,rows

def secant(f,x0,x1,tol,max_iter=100):
    rows=[]
    for i in range(1,max_iter+1):
        f0,f1=f(x0),f(x1)
        if f1-f0==0: raise ZeroDivisionError("Zero denominator in Secant iteration.")
        x2=x1 - f1*(x1-x0)/(f1-f0)
        err=abs(x2-x1)
        rows.append((i,x2,f(x2),err))
        if abs(f(x2))<tol or err<tol: return x2,err,i,rows
        x0,x1=x1,x2
    return x2,err,max_iter,rows

def newton_raphson(f,x0,tol,max_iter=100):
    rows=[]; x=x0
    for i in range(1,max_iter+1):
        fx=f(x); dfx=numerical_derivative(f,x)
        if dfx==0: raise ZeroDivisionError("Zero derivative encountered.")
        x_new=x-fx/dfx; err=abs(x_new-x)
        rows.append((i,x_new,f(x_new),err))
        if abs(f(x_new))<tol or err<tol: return x_new,err,i,rows
        x=x_new
    return x,err,max_iter,rows

def fixed_point_iteration(g,x0,tol,max_iter=100):
    rows=[]; x=x0
    for i in range(1,max_iter+1):
        x_new=g(x); err=abs(x_new-x)
        rows.append((i,x_new,None,err))
        if err<tol: return x_new,err,i,rows
        x=x_new
    return x,err,max_iter,rows

def modified_secant(f,x0,delta,tol,max_iter=100):
    rows=[]; x=x0
    for i in range(1,max_iter+1):
        fx=f(x)
        dfx_approx=(f(x+delta*x)-fx)/(delta*x) if x!=0 else numerical_derivative(f,x)
        if dfx_approx==0: raise ZeroDivisionError("Zero derivative approximation.")
        x_new=x-fx/dfx_approx; err=abs(x_new-x)
        rows.append((i,x_new,f(x_new),err))
        if abs(f(x_new))<tol or err<tol: return x_new,err,i,rows
        x=x_new
    return x,err,max_iter,rows

def get_float(prompt, default=None):
    while True:
        try:
            s=input(f"{prompt}{' ['+str(default)+']' if default else ''}: ")
            if s.strip()=="" and default is not None: return float(default)
            return float(s)
        except: print("Invalid number.")

def main():
    print("ZOF CLI — Zero of Functions")
    expr=input("f(x) = ")
    f=parse_function(expr)

    print("Choose method:")
    methods=["Bisection","Regula Falsi","Secant","Newton-Raphson","Fixed Point","Modified Secant"]
    for i,m in enumerate(methods,1): print(i,m)
    choice=int(get_float("Method number",1))

    tol=get_float("Tolerance",1e-6)
    max_iter=int(get_float("Max iterations",100))

    if choice==1:
        a=get_float("a"); b=get_float("b")
        root,err,it,rows=bisection(f,a,b,tol,max_iter)
    elif choice==2:
        a=get_float("a"); b=get_float("b")
        root,err,it,rows=regula_falsi(f,a,b,tol,max_iter)
    elif choice==3:
        x0=get_float("x0"); x1=get_float("x1")
        root,err,it,rows=secant(f,x0,x1,tol,max_iter)
    elif choice==4:
        x0=get_float("x0")
        root,err,it,rows=newton_raphson(f,x0,tol,max_iter)
    elif choice==5:
        gexpr=input("g(x)="); g=parse_function(gexpr)
        x0=get_float("x0")
        root,err,it,rows=fixed_point_iteration(g,x0,tol,max_iter)
    elif choice==6:
        x0=get_float("x0"); delta=get_float("delta",1e-3)
        root,err,it,rows=modified_secant(f,x0,delta,tol,max_iter)

    print_iteration_table(rows)
    print("Root:",root," Error:",err," Iter:",it)

if __name__ == "__main__":
    main()
