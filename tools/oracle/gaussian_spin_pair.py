# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Independent Decimal reference for a four-centre Gaussian triplet model.

This validation tool solves the two-alpha-electron UHF equations for four unit
nuclear charges with one normalized s primitive of exponent 0.7 per centre.
It evaluates the direct electron spin-spin tensor with explicit Coulomb
integrals. It is a finite-basis mathematical reference, not a complete SOC D
model, a production SCF solver or an experimental molecular prediction.

The CLI reads a JSON object with positions_bohr and optional precision_digits
from stdin and writes its generated reference to stdout. Raw reference results
belong outside the distributable source. Only the Python standard library is
required; no production integral, response or SCF routines are imported.
"""
from decimal import Decimal as D, localcontext
import json
import sys
import time

PI='3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'

def zeros(n):return [[D(0) for _ in range(n)] for _ in range(n)]
def transpose(a):return list(map(list,zip(*a)))
def mm(a,b):return [[sum((x*y for x,y in zip(row,column)),D(0)) for column in zip(*b)] for row in a]
def eye(n):return [[D(int(i==j)) for j in range(n)] for i in range(n)]
def scale(a,x):return [[x*v for v in row] for row in a]
def add(a,b):return [[x+y for x,y in zip(row,other)] for row,other in zip(a,b)]
def dot(a,b):return sum((x*y for row,other in zip(a,b) for x,y in zip(row,other)),D(0))
def largest(a):return max(abs(v) for row in a for v in row)


def eigen(a,tolerance):
    a=[row[:] for row in a];n=len(a);vectors=eye(n)
    for _ in range(2000):
        p,q=max(((i,j) for i in range(n) for j in range(i+1,n)),key=lambda ij:abs(a[ij[0]][ij[1]]))
        off=a[p][q]
        if abs(off)<tolerance:break
        tau=(a[q][q]-a[p][p])/(2*off)
        t=(D(1) if tau>=0 else D(-1))/(abs(tau)+(1+tau*tau).sqrt())
        c=1/(1+t*t).sqrt();s=t*c
        app,aqq=a[p][p],a[q][q]
        a[p][p]=app-t*off;a[q][q]=aqq+t*off;a[p][q]=a[q][p]=D(0)
        for i in range(n):
            if i in (p,q):continue
            u,v=a[i][p],a[i][q]
            a[i][p]=a[p][i]=c*u-s*v
            a[i][q]=a[q][i]=s*u+c*v
        for row in vectors:
            u,v=row[p],row[q];row[p]=c*u-s*v;row[q]=s*u+c*v
    else:raise RuntimeError('independent Jacobi solver did not converge')
    order=sorted(range(n),key=lambda i:a[i][i])
    return [a[i][i] for i in order],[[row[i] for i in order] for row in vectors]


def linear(matrix,rhs,tolerance):
    n=len(rhs);a=[row[:]+[value] for row,value in zip(matrix,rhs)]
    reference=max(abs(v) for row in matrix for v in row)
    for i in range(n):
        pivot=max(range(i,n),key=lambda j:abs(a[j][i]))
        if abs(a[pivot][i])<reference*tolerance:raise ArithmeticError('rank deficient Pulay history')
        a[i],a[pivot]=a[pivot],a[i];d=a[i][i];a[i]=[v/d for v in a[i]]
        for j in range(n):
            if j==i:continue
            f=a[j][i];a[j]=[u-f*v for u,v in zip(a[j],a[i])]
    return [row[-1] for row in a]


def boys(n,t,tolerance):
    term=D(1);value=term/D(2*n+1)
    for k in range(1,2000):
        term*=-t/D(k);increment=term/D(2*n+2*k+1);value+=increment
        if abs(increment)<tolerance:return value
    raise RuntimeError('independent Coulomb series did not converge')


def integrals(r,digits):
    n=4;a=D.from_float(.7);pi=D(PI);p=2*a;mu=a/2
    norm=(2*a/pi).sqrt().sqrt()**3
    tol=D(10)**(-digits-3)
    pairs={};overlap=zeros(n);core=zeros(n)
    for i in range(n):
        for j in range(n):
            r2=sum(((u-v)**2 for u,v in zip(r[i],r[j])),D(0))
            attenuation=(-mu*r2).exp();center=[(u+v)/2 for u,v in zip(r[i],r[j])]
            overlap[i][j]=attenuation
            kinetic=mu*(3-2*mu*r2)*attenuation
            attraction=D(0)
            for nucleus in r:
                distance=sum(((u-v)**2 for u,v in zip(center,nucleus)),D(0))
                attraction-=2*pi/p*norm*norm*attenuation*boys(0,p*distance,tol)
            core[i][j]=kinetic+attraction;pairs[i,j]=(attenuation,center)
    eri={};spin={};rho=p/2
    prefactor=2*pi*pi*pi.sqrt()/(p*p*(2*p).sqrt())*norm**4
    for i in range(n):
        for j in range(n):
            ab,pc=pairs[i,j]
            for k in range(n):
                for l in range(n):
                    cd,qc=pairs[k,l];delta=[u-v for u,v in zip(pc,qc)]
                    t=rho*sum((v*v for v in delta),D(0));weight=prefactor*ab*cd
                    f0,f1,f2=(boys(m,t,tol) for m in range(3))
                    eri[i,j,k,l]=weight*f0
                    spin[i,j,k,l]=[weight*(2*rho*D(int(x==y))*f1-4*rho*rho*delta[x]*delta[y]*f2)
                        for x in range(3) for y in range(3)]
    repulsion=sum((1/sum(((u-v)**2 for u,v in zip(r[i],r[j])),D(0)).sqrt()
        for i in range(n) for j in range(i)),D(0))
    return overlap,core,eri,spin,repulsion


def calculate(r,digits,seed=None):
    with localcontext() as context:
        context.prec=digits
        started=time.monotonic();n=4
        s,h,eri,spin,enuc=integrals(r,digits)
        eig_tol=D(10)**(-digits+12);scf_tol=D(10)**(-digits+22)
        se,sv=eigen(s,eig_tol);assert min(se)>0
        x=mm([[value/e.sqrt() for value,e in zip(row,se)] for row in sv],transpose(sv))
        def density(f):
            energies,c=eigen(mm(mm(x,f),x),eig_tol);c=mm(x,c)
            return [[sum((c[i][k]*c[j][k] for k in range(2)),D(0)) for j in range(n)] for i in range(n)],energies
        def fock(p):
            return [[h[i][j]+sum((p[k][l]*(eri[i,j,k,l]-eri[i,k,j,l])
                for k in range(n) for l in range(n)),D(0)) for j in range(n)] for i in range(n)]
        p=[[D(v) for v in row] for row in seed] if seed else density(h)[0]
        history=[];energies=None
        for iteration in range(300):
            f=fock(p);fps=mm(mm(f,p),s);spf=mm(mm(s,p),f)
            residual=add(fps,scale(spf,D(-1)));norm=largest(residual)
            if norm<scf_tol:break
            history.append((f,residual));history=history[-6:]
            trial=f
            while len(history)>1:
                m=len(history)
                b=[[dot(ei,ej) for _,ej in history]+[D(-1)] for _,ei in history]
                b.append([D(-1)]*m+[D(0)])
                try:coefficients=linear(b,[D(0)]*m+[D(-1)],D(10)**(-digits+15))[:-1]
                except ArithmeticError:history.pop(0);continue
                trial=zeros(n)
                for coefficient,(old_f,_) in zip(coefficients,history):trial=add(trial,scale(old_f,coefficient))
                break
            p,energies=density(trial)
        else:raise RuntimeError('independent SCF did not converge')
        f=fock(p);canonical,energies=density(f)
        assert largest(add(p,scale(canonical,D(-1))))<scf_tol*100
        trace=dot(p,s);idempotency=largest(add(mm(mm(p,s),p),scale(p,D(-1))))
        assert abs(trace-2)<scf_tol*100 and idempotency<scf_tol*100
        electronic=dot(p,add(h,f))/2;energy=electronic+enuc
        value=[D(0)]*9
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        weight=p[i][j]*p[k][l]-p[i][k]*p[j][l]
                        for component in range(9):value[component]+=weight*spin[i,j,k,l][component]
        alpha=7.2973525693e-3;ge=2.00231930436256
        factor=D.from_float((alpha*alpha)*(ge*ge)/16.0)
        value=[v*factor for v in value]
        return dict(energy=str(energy),electronic_energy=str(electronic),nuclear_repulsion=str(enuc),
            density=[[str(v) for v in row] for row in p],spin_spin=[str(v) for v in value],
            iterations=iteration+1,stationarity=str(norm),particle_number=str(trace),
            idempotency=str(idempotency),gap=str(energies[2]-energies[1]),
            raw_D_trace=str(value[0]+value[4]+value[8]),seconds=time.monotonic()-started,
            precision_digits=digits,positions=[[str(v) for v in row] for row in r])



def reference(positions_bohr, precision_digits=70, seed_density=None):
    if not isinstance(precision_digits, int) or isinstance(precision_digits, bool) or not 50 <= precision_digits <= 90:
        raise ValueError("reference precision must be an integer from 50 through 90")
    if len(positions_bohr) != 4 or any(len(row) != 3 for row in positions_bohr):
        raise ValueError("reference requires four three-dimensional centres")
    def number(value):
        return D.from_float(value) if isinstance(value, float) else D(value)
    with localcontext() as context:
        context.prec = precision_digits
        coordinates = [[number(value) for value in row] for row in positions_bohr]
        if any(not value.is_finite() for row in coordinates for value in row):
            raise ValueError("reference positions must be finite")
        return calculate(coordinates, precision_digits, seed_density)


if __name__ == "__main__":
    request = json.load(sys.stdin)
    result = reference(request["positions_bohr"], request.get("precision_digits", 70),
                       request.get("seed_density"))
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
