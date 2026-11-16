# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Independent second-order propagator oracle in a complete determinant space.

The reference contractions are direct creation/annihilation matrices, not
the production pair-density, integral-rotation or response contractions.
NumPy is used only to freeze oracle data, never by the package at runtime.
Pass an explicit output path; no local paths enter the oracle data.
"""
from itertools import combinations
from pathlib import Path
import argparse
import json
import numpy as np

orbitals=4
occupied=2
basis=[tuple(a)+tuple(x+orbitals for x in b)
       for a in combinations(range(orbitals),occupied)
       for b in combinations(range(orbitals),occupied)]
index={state:k for k,state in enumerate(basis)}
dimension=len(basis)
reference=index[(0,1,4,5)]
vacuum=np.eye(dimension)[:,reference]

def replacement(p,q):
    result=np.zeros((dimension,dimension))
    for j,state in enumerate(basis):
        if q not in state:continue
        values=list(state);a=values.index(q);values.pop(a)
        if p in values:continue
        b=sum(v<p for v in values);values.insert(b,p)
        result[index[tuple(values)],j]=(-1.)**(a+b)
    return result

one={(s,p,q):replacement(s*orbitals+p,s*orbitals+q)
     for s in range(2) for p in range(orbitals) for q in range(orbitals)}
rng=np.random.default_rng(23)
factors=rng.normal(size=(6,orbitals,orbitals))
factors=0.04*(factors+factors.transpose(0,2,1))
g=np.einsum("kpq,krs->pqrs",factors,factors)
interaction=np.zeros((dimension,dimension))
mean=np.zeros_like(interaction)
for s in range(2):
    for t in range(2):
        for p in range(orbitals):
            for q in range(orbitals):
                for r in range(orbitals):
                    for u in range(orbitals):
                        operator=one[s,p,q]@one[t,r,u]
                        if s==t and q==r:operator=operator-one[s,p,u]
                        interaction+=0.5*g[p,q,r,u]*operator
                coefficient=sum(g[p,q,j,j]-(g[p,j,j,q] if s==t else 0.)
                                for j in range(occupied))
                mean+=coefficient*one[s,p,q]
constant=interaction[reference,reference]
residual=interaction-mean+constant*np.eye(dimension)
energies=np.array([-1.3,-0.8,0.4,0.9,-1.1,-0.6,0.5,1.2])
gaps=np.array([sum(energies[list(state)])-sum(energies[list(basis[reference])]) for state in basis])
free=np.diag(gaps)
singles=[]
single_indices=[]
for s in range(2):
    for a in range(occupied,orbitals):
        for i in range(occupied):
            operator=one[s,a,i]
            singles.append(operator)
            single_indices.append(np.flatnonzero(operator@vacuum)[0])
doubles=[]
for s,t in [(0,0),(0,1),(1,1)]:
    for i in range(occupied):
        for a in range(occupied,orbitals):
            for j in range(occupied):
                for b in range(occupied,orbitals):
                    if s==t and (i>=j or a>=b):continue
                    doubles.append(one[s,a,i]@one[t,b,j])
assert np.max(np.abs(residual@vacuum)[single_indices])<1e-15
first=np.zeros(dimension)
nonreference=np.arange(dimension)!=reference
first[nonreference]=-(residual@vacuum)[nonreference]/gaps[nonreference]
second_singles=np.zeros(dimension)
second_singles[single_indices]=-(residual@first)[single_indices]/gaps[single_indices]
norm=first@first

def comm(a,b):return a@b-b@a

def term2(operator,extra=None):
    result=first@operator@first-norm*(vacuum@operator@vacuum)
    if extra is not None:result+=first@extra@vacuum+vacuum@extra@first
    return result


operators=[]
one_particle=[]
for k in range(3):
    for imaginary in [False,True]:
        value=np.zeros((dimension,dimension),dtype=complex)
        matrices=[]
        for s in range(2):
            matrix=rng.normal(size=(orbitals,orbitals))
            raw=(matrix-matrix.T if imaginary else matrix+matrix.T)*0.3
            matrices.append(raw.tolist())
            for i in range(orbitals):
                for j in range(orbitals):value+=(1j if imaginary else 1.)*raw[i,j]*one[s,i,j]
        operators.append(value)
        one_particle.append(matrices)

requests=[]
def add(kind,i,j,order,target,e0,e1=None,imaginary=False):
    requests.append((kind,i,j,order,target,e0,e1,imaginary))
for i,qi in enumerate(singles):
    for j,qj in enumerate(singles):
        e0=comm(qi.T,comm(free,qj));e1=comm(qi.T,comm(residual,qj))
        add("A",i,j,2,2.*term2(e0,e1),e0,e1)
        e0=-comm(qi.T,comm(free,qj.T));e1=-comm(qi.T,comm(residual,qj.T))
        add("B",i,j,2,2.*term2(e0,e1),e0,e1)
        metric=comm(qi.T,qj)
        if np.any(metric):add("S",i,j,2,2.*term2(metric),metric)
    for j,qd in enumerate(doubles):
        e0=comm(qi.T,comm(free,qd));e1=comm(qi.T,comm(residual,qd))
        add("C",i,j,1,vacuum@e1@vacuum,e0,e1)
    for j,operator in enumerate(operators):
        observable=comm(qi.T,operator)
        target=term2(observable)+second_singles@observable@vacuum+vacuum@observable@second_singles
        add("P",i,j,2,2.*target,observable,imaginary=bool(j%2))
for i,qi in enumerate(doubles):
    for j,operator in enumerate(operators):
        observable=comm(qi.T,operator)
        target=first@observable@vacuum+vacuum@observable@first
        add("D",i,j,1,target,observable,imaginary=bool(j%2))

def exact(parameter):
    _,vectors=np.linalg.eigh(free+parameter*residual)
    state=vectors[:,0]
    values=[]
    for _,_,_,_,_,a,b,imag in requests:
        value=state@(a+(parameter*b if b is not None else 0.))@state
        values.append(float(value.imag if imag else value.real))
    return np.array(values)

intervals=[[0.16,0.12,0.08,0.06],[0.04,0.03,0.02,0.015],[0.32,0.24,0.16,0.12]]
zero=exact(0.)
estimates=[]
orders=np.array([r[3] for r in requests])
for steps in intervals:
    values=[]
    for h in steps:
        plus=exact(h);minus=exact(-h)
        fd=np.where(orders==1,(plus-minus)/(2*h),(plus+minus-2*zero)/(h*h))
        values.append(fd.tolist())
    estimates.append(values)
data={
    "provenance":"Independent full fixed-Sz determinant Hamiltonian; exact symmetric diagonalization at each signed interaction strength; numpy "+np.__version__,
    "seed":23,"orbitals":orbitals,"occupied":occupied,"integrals":g.reshape(-1).tolist(),
    "energies":energies.reshape(2,orbitals).tolist(),"operators":one_particle,
    "requests":[{"kind":kind,"row":i,"column":j,"order":order,"target":float(target.imag if imag else np.real(target))} for kind,i,j,order,target,_,_,imag in requests],
    "intervals":intervals,"estimates":estimates,
}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("output",type=Path)
args=parser.parse_args()
args.output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
print(f"Wrote {len(requests)} independent derivative coordinates to {args.output.name}")
