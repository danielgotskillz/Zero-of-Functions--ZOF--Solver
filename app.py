from flask import Flask, render_template, request
import math, traceback

app = Flask(__name__)

SAFE_MATH={k:getattr(math,k) for k in dir(math) if not k.startswith("__")}
SAFE_MATH.update({"abs":abs,"min":min,"max":max})

def parse_function(expr):
    def f(x):
        return eval(expr,{"__builtins__":{}},dict(SAFE_MATH,x=x))
    return f

def numerical_derivative(f,x,h=1e-6):
    return (f(x+h)-f(x-h))/(2*h)

def run_method(method,fexpr,params):
    f=parse_function(fexpr)
    max_iter=int(params.get("max_iter",100))
    tol=float(params.get("tol",1e-6))
    rows=[]; root=None
    try:
        if method=="bisection":
            a=float(params["a"]); b=float(params["b"])
            fa,fb=f(a),f(b)
            if fa*fb>0: raise ValueError("Opposite signs needed.")
            for i in range(1,max_iter+1):
                c=(a+b)/2; fc=f(c); err=abs(b-a)/2
                rows.append((i,c,fc,err))
                if abs(fc)<tol or err<tol: root=c; break
                if fa*fc<0: b=c; fb=fc
                else: a=c; fa=fc
        elif method=="regula":
            a=float(params["a"]); b=float(params["b"])
            fa,fb=f(a),f(b)
            if fa*fb>0: raise ValueError("Opposite signs needed.")
            c=a
            for i in range(1,max_iter+1):
                c_prev=c
                c=(a*fb-b*fa)/(fb-fa)
                fc=f(c); err=abs(c-c_prev) if i>1 else float("inf")
                rows.append((i,c,fc,err))
                if abs(fc)<tol or err<tol: root=c; break
                if fa*fc<0: b=c; fb=fc
                else: a=c; fa=fc
        elif method=="secant":
            x0=float(params["x0"]); x1=float(params["x1"])
            for i in range(1,max_iter+1):
                f0,f1=f(x0),f(x1)
                if f1-f0==0: raise ZeroDivisionError("Zero denom")
                x2=x1 - f1*(x1-x0)/(f1-f0)
                err=abs(x2-x1)
                rows.append((i,x2,f(x2),err))
                if abs(f(x2))<tol or err<tol: root=x2; break
                x0,x1=x1,x2
        elif method=="newton":
            x=float(params["x0"])
            for i in range(1,max_iter+1):
                fx=f(x); dfx=numerical_derivative(f,x)
                if dfx==0: raise ZeroDivisionError("Zero derivative")
                x_new=x-fx/dfx; err=abs(x_new-x)
                rows.append((i,x_new,f(x_new),err))
                if abs(f(x_new))<tol or err<tol: root=x_new; break
                x=x_new
        elif method=="fixed":
            g=parse_function(params["g"])
            x=float(params["x0"])
            for i in range(1,max_iter+1):
                x_new=g(x); err=abs(x_new-x)
                rows.append((i,x_new,None,err))
                if err<tol: root=x_new; break
                x=x_new
        elif method=="modified":
            x=float(params["x0"]); delta=float(params.get("delta",1e-3))
            for i in range(1,max_iter+1):
                fx=f(x)
                dfx_approx=(f(x+delta*x)-fx)/(delta*x) if x!=0 else numerical_derivative(f,x)
                if dfx_approx==0: raise ZeroDivisionError("Zero derivative approx")
                x_new=x-fx/dfx_approx; err=abs(x_new-x)
                rows.append((i,x_new,f(x_new),err))
                if abs(f(x_new))<tol or err<tol: root=x_new; break
                x=x_new
    except Exception as e:
        return {"error":str(e)+"\n"+traceback.format_exc()}

    return {"rows":rows,"root":root}

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        method=request.form["method"]
        fexpr=request.form["fexpr"]
        params=request.form.to_dict()
        result=run_method(method,fexpr,params)
        return render_template("index.html",result=result,params=params,method=method,fexpr=fexpr)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
