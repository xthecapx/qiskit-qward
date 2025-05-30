COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
KAIFENG BU1, ROY J. GARCIA1, ARTHUR JAFFE1, DAX ENSHAN KOH2, AND
ABSTRACT. Quantum circuit complexity—a measure of the minimum num- ber of gates needed to implement a given unitary transformation—is a funda- mental concept in quantum computation, with widespread applications ranging from determining the running time of quantum algorithms to understanding the physics of black holes. In this work, we study the complexity of quantum circuits using the notions of sensitivity, average sensitivity (also called influ- ence), magic, and coherence. We characterize the set of unitaries with vanish- ing sensitivity and show that it coincides with the family of matchgates. Since matchgates are tractable quantum circuits, we have proved that sensitivity is necessary for a quantum speedup. As magic is another measure to quantify quantum advantage, it is interesting to understand the relation between magic and sensitivity. We do this by introducing a quantum version of the Fourier entropy-influence relation. Our results are pivotal for understanding the role of sensitivity, magic, and coherence in quantum computation.
CONTENTS
1. Introduction
2. Sensitivity and circuit complexity
3. Quantum Fourier entropy and influence
4. Magic and circuit complexity
5. Coherence and circuit complexity
6. Concluding remarks
Acknowledgments
LU LI3
2
                                                               5
                                                              18
                                                              23
                                                              28
                                                              30
                                                              31
                                                              31
                                                              36
37 38
Email:
(2) Institute of High Performance Computing, Agency for Science, Technology and Re- search (A*STAR), 1 Fusionopolis Way, #16-16 Connexis, Singapore 138632, Singapore. Email: dax_koh@ihpc.a-star.edu.sg
(3) Department of Mathematics, Zhejiang Sci-Tech University, Hangzhou, Zhejiang 310018, China. Email: lilu93@zju.edu.cn
1
Appendix A. Appendix B. Appendix C.
References
OTOCs
Boolean Fourier entropy-influence conjecture
Discrete Wigner function and symplectic Fourier transformation
(1) Harvard
kfbu@fas.harvard.edu; roygarcia@g.harvard.edu; arthur_jaffe@harvard.edu
University, Cambridge,
Massachusetts
02138,
USA.
arXiv:2204.12051v1 [quant-ph] 26 Apr 2022
 
2 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
1. INTRODUCTION
A central problem in the field of quantum information and computation is to compute the complexity required to implement a target unitary operation U . One usually defines this to be the minimum number of basic gates needed to synthe- size U from some initial fiducial state [1–3]. To determine the so-called quantum circuit complexity of a given unitary operation, a closely related concept, called the circuit cost, was proposed and investigated in a series of seminal papers by Nielsen et al. [4–7]. Surprisingly, the circuit cost, defined as the minimal geodesic distance between the target unitary operation and the identity operation in some curved geometry, was shown to provide a useful lower bound for the quantum circuit complexity [5, 6].
In more recent years, the quantum circuit complexity, as well as the circuit cost, was shown to also play an important role in the domain of high-energy physics [8–12]. For example, its evolution was found to exhibit identical pat- terns to how the geometry hidden inside black hole horizons evolves. Further studies have also investigated the circuit complexity in the context of quantum field theories [13–15], including conformal field theory [16, 17] and topological quantum field theory [18]. Recently, Brown and Susskind argue that the prop- erty of possessing less-than-maximal entropy, or uncomplexity, could be thought of as a resource for quantum computation [8]. This was supported by Yunger Halpern et al. who present a resource theory of quantum uncomplexity [19]. Fur- thermore, a connection between quantum entanglement and quantum circuit com- plexity was revealed by Eisert, who proved that the entangling power of a unitary transformation provides a lower bound for its circuit cost [20].
Let us summarize the main ideas we present in this paper, which we will de- scribe in more detail in §1.1. In this paper, we study the quantum circuit com- plexity of quantum circuits via their sensitivities, magic, and coherence. The first property, namely sensitivity, is a measure of complexity that plays an important role in the analysis of Boolean functions [21, 22] and can be applied to a range of topics, including the circuit complexity of Boolean circuits [23–25], error- correcting codes [26], and quantum query complexity [27]. A fundamental result in circuit complexity is that the average sensitivity, also called the influence, of constant-depth Boolean circuits is bounded above by the depth and the number of gates in the circuit [23, 24]. While the notion of influence has been general- ized to describe quantum Boolean functions [28], considerably little is hitherto known about the connection between the sensitivity (or influence) and the cir- cuit complexity of a quantum circuit. In this regard, our first result provides an upper bound on the circuit sensitivity—a measure of sensitivity for unitary transformations—of a quantum circuit by its circuit cost.
Secondly, we characterize unitaries with zero circuit sensitivity, which we call stable unitaries. We generalize the definition of sensitivity to Clifford algebras, where we use the noise operator defined by Carlen and Lieb [29]. We find that stable gates in this case are exactly matchgates, a well-known family of tractable quantum circuits [30–36]. This provides a new understanding of matchgates via sensitivity. Our result also implies that sensitivity is necessary for a quantum

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 3
computational advantage; for a more extended discussion, see Remark 2. In addi- tion, we show a relation between average scrambling and the average sensitivity. Magic is another important resource in quantum computation, which character- izes how far away a quantum state (or gate) is from the set of stabilizer states (or gates). The Gottesman-Knill theorem [37] states that stabilizer circuits compris- ing Clifford unitaries and stabilizer inputs and measurements can be simulated efficiently on a classical computer. Hence, magic is necessary to realize a quan- tum advantage [38–42]. Magic measures have been used to bound the classical simulation time in quantum computation [43–50], and also in condensed matter physics [51]. However, the relationship between magic and the complexity of
quantum circuits has so far largely been unexplored.
To reveal the connection between magic and circuit complexity, we implement
two different approaches. The first approach (see §4.1) uses consequences of the quantum Fourier entropy-influence relation and conjecture, which shows the rela- tion between magic and sensitivity. It can be summarized by the set of inferences diagrammed here:
magic-sensitivity relation + sensitivity-complexity relation
QEFI magic-complexity relation
Depending on whether one takes a proven result or a conjectured bound, one arrives at an uninteresting or interesting result, respectively. The classical Fourier entropy-influence conjecture was proposed by Friedgut and Kalai [52], and has many useful implications in the analysis of Boolean functions and computational learning theory. For example, if the Fourier entropy-influence conjecture holds, then it implies the existence of a polynomial-time agnostic learning algorithm for disjunctive normal forms (DNFs) [53].
The second method (see §4.2) we take here is to exhibit the connection between magic and circuit cost directly by introducing the magic rate and magic power. Magic power quantifies the incremental magic by the circuit, while the magic rate quantifies the small incremental magic in infinitesimal time.
Finally, we show the connection between coherence and circuit complexity for quantum circuits. Quantum coherence, which arises from superposition, plays a fundamental role in quantum mechanics. The recent significant developments in quantum thermodynamics [54, 55] and quantum biology [56–58] have shown that coherence can be a very useful resource at the nanoscale. This has led to the development of the resource theory of coherence [59–64]. However, thus far, little is known about the connection between coherence and circuit complexity. In this paper, we address this gap and provide a lower bound on the circuit cost by the power of coherence in the circuit.
The rest of the paper is structured as follows. In §1.1, we summarize the main results of our work. In §2, we investigate the connection between circuit com- plexity and circuit sensitivity and propose a new interpretation of matchgates in terms of sensitivity. In §3, we consider the relationship between quantum Fourier

4 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
entropy and influence. In §4, we study the connection between magic and the cir- cuit cost of quantum circuits. In §5, we study the connection between coherence and the circuit cost of quantum circuits.
1.1. Main results. We start by summarizing three of our main results concern- ing lower bounds on quantum circuit complexity in terms of average sensitivity, magic, and coherence. Here, the complexity of a quantum circuit is taken to be the circuit cost introduced by Nielsen et al.:
Definition 1 (Nielsen et al. [6]). Let U ∈ SU(dn) be a unitary operation and h1,...,hm be traceless Hermitian operators that are supported on 2 qudits and normalized as ∥hi∥∞ = 1. The circuit cost of U, with respect to h1,...,hm, is defined as
Z1m
Cost(U ) := inf ∑ |r j (s)|ds. (1)
0 j=1
where the infimum above is taken over all continuous functions r j : [0, 1] → R
satisfying
and
Z1
U=Pexp −i H(s)ds , (2)
0
m
H(s)= ∑rj(s)hj, (3)
j=1 where P denotes the path-ordering operator.
The theorem below, which gives lower bounds for the circuit cost, collects Theorems 12, 43 and 50 in one place:
Theorem 2 (Results on Circuit Complexity). The circuit cost of a quantum circuit U ∈ SU(dn) is lower bounded as follows:
Cost(U)≥cmaxCiS[U],M[U],Cr(U), (4) d2 log(d)
where c is a universal constant independent of d and n. The quantities CiS[U] (M[U], Cr(U), respectively), defined formally in (13) ((51), (64), respectively), quantify the sensitivity (magic, coherence, respectively) of quantum circuits. Note that here and throughout this paper, the logarithm is taken to be of base 2.
We also define the circuit sensitivity CiSG for any unitary in terms of the gen- erators of the Clifford algebra, yielding a new understanding for matchgates (see Theorem 20 for more details):
Theorem 3 (Matchgates via Sensitivity). A unitary U satisfies CiSG[U] = 0 if and only if it is a matchgate.
Matchgates are a well-known family of tractable circuits, and our result shows that CiSG could also be used to serve as a measure of non-Gaussianity (noting that matchgates are also called Gaussian operations).
  
COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 5
To show the connection between magic and influence (or non-Gaussianity quantified by influence), we also prove the following statement (an informal ver- sion of Theorem 23):
Theorem 4 (Quantum Fourier Entropy-Influence Relation). For any linear n-qudit operator O with ∥O∥2 = 1, we have
H[O] ≤ c[logn+logd]I[O]+h[PO[⃗0]],
where h(x) := −xlogx−(1−x)log(1−x) is the binary entropy and c is a uni-
versal constant.
2. SENSITIVITY AND CIRCUIT COMPLEXITY
Given the n-qudit system H = (Cd)⊗n, the inner product between two op-
1†
erators A and B on H is defined as ⟨A,B⟩ = dn Tr A B , and the l2 norm in-
A†A. Taking ( 5 )
p 
  duced by the inner product is defined by ∥A∥ := ⟨A,A⟩. More generally, for
2
p ≥ 1, the l norm is defined as ∥A∥ = ( 1 Tr [|A|p])1/p with |A| =
√
 p pdn
V := Zd × Zd , the set of generalized Pauli operators is
 P n = { P ⃗a : P ⃗a = ⊗ i P a i } ⃗a ∈ V n ,
where Pai = XsiZti for any ai = (si,ti) ∈ V. Here, the qudit Pauli X and Z are the shift and clock operators, respectively, defined by X | j⟩ = | j + 1 (mod d)⟩ and Z|j⟩=exp(2ijπ/d)|j⟩,respectively.LetusdefinePO[⃗a]forany⃗a∈Vn as
PO[⃗a]= 1|Tr[OP⃗a]|2,∀⃗a∈Vn. (6) d2n
Note that the condition ∥O∥2 = 1 is equivalent to saying that {PO[⃗a]}⃗a is a prob- ability distribution over V n .
2.1. Influence.
Definition 5 (Montanaro and Osborne [28]). Given a linear operator O, the local
 influence at the j-th qudit is defined as
I j [ O ] = ∑ P O [ ⃗a ] ,
⃗a:a j ̸=(0,0)
and the total influence is defined as the sum of all the local influences:
I[O]= ∑Ij[O]. j∈[n]
( 7 )
(8)
With the assumption that PO in (6) is a probability distribution, the local influ- ence and total influence can be rewritten, respectively, as
Ij[O] = ∑ PO[⃗a]= E |aj|,
where |a j | = 1 if a j = (0, 0) and 0 otherwise; supp(⃗a) (the support of ⃗a) denotes the set of indices i for which ai ̸= 0; and |⃗a| := |supp(⃗a)|.
⃗a∈PO I[O] = ∑ |supp(⃗a)|PO[⃗a] = E
(9) |⃗a|, (10)
⃗a:a j ̸=(0,0)
⃗a ∈ V n ⃗a ∼ P O

6 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
Note that it is easy to see that the influence can be used to quantify the sensi- tivity of the single-qudit depolarizing channel Dγ (·) = (1 − γ )(·) + γ Tr [·] I/d as follows
∂ D(j)[O] 2 =−2I [O], (11) ∂γγ 2γ=0 j
where D( j) denotes the depolarizing channel acting on the j-th qudit. This implies γ
that
∂ D⊗n[O] 2 =−2I[O]. (12) ∂γγ 2γ=0
Hence, influence is an average version of sensitivity with respect to depolarizing noise. Note that the notion of influence, Ij(O) and I(O), could be applied to quantum states |ψ⟩ by setting O = √dn|ψ⟩⟨ψ| to ensure that the corresponding probability distribution PO defined in (6) sums to 1.
2.2. Circuit sensitivity and complexity.
Definition 6 (Circuit Sensitivity). For a unitary U , the circuit sensitivity CiS[U ]
is the change of influence caused by U , defined as
CiS[U]= max I[UOU†]−I[O] . (13)
O:∥O∥2 =1
First, let us present a basic lemma of circuit sensitivity, which indicates that in
the maximization in (13), it suffices to just consider traceless operators:
Lemma 7. The circuit sensitivity equals
CiS[U]= max I[UOU†]−I[O] , (14)
O:∥O∥2=1,Tr[O]=0
that is, it suffices to just consider a maximization over all traceless operators with
∥O∥2 = 1.
P r o o f . F i r s t , P O [ ⃗0 ] d e fi n e d i n ( 6 ) i s u n i t a r i l y i n v a r i a n t . H e n c e , i f T r [ O ] ̸ = 0 , l e t u s
   define a new operator O′ as
′ 1Tr[O]
O=q O− dn I . 1 − P O [ ⃗0 ]
   Then O′ satisfies the conditions Tr [O′] = 0 and ∥O′∥2 = 1. Also,
I[O′] = I[UO′U†] =
1 I[O], 1 − P O [ ⃗0 ]
1 I[UOU†]. 1 − P O [ ⃗0 ]
  Hence, we have
I[UO′U†]−I[O′] =
1 (I[UOU†]−I[O]). 1 − P O [ ⃗0 ]
 Therefore, the maximum must be attained by traceless operators. 

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 7
Now, let us consider the n-qudit Hamiltonian acting nontrivially on a k-qudit subsystem. We prove here a simple upper bound on the total change of the total influence I through unitary evolution.
Proposition 8 (Small Total Circuit Sensitivity). Given an n-qudit system with a Hamiltonian H acting nontrivially on a k-qudit subsystem, the total change of influence induced by the unitary Ut = e−itH is bounded from above by k:
CiS[Ut ] ≤ k. (15)
Proof. Since H acts on only a k-qudit subsystem, there exists a subset S of size k such that H = HS ⊗ ISc and Ut = US ⊗ ISc . Due to the subadditivity of the circuit sensitivity under tensorization (Proposition 14), CiS[Ut] ≤ CiS[US] ≤ k. 
Now, let us introduce the influence rate to quantify the change of influence in an infinitesimally small time interval. This will be used to prove the connection between circuit sensitivity and circuit complexity.
Definition 9 (Influence Rate). Given an n-qudit Hamiltonian H and a linear operator O with ∥O∥2 = 1, the influence rate of the unitary Ut = e−itH acting on O is defined as follows
RI(H,O) = dI[UtOUt†] , (16) dt t=0
which can be used to quantify small incremental influence for a given unitary evolution.
By a direct calculation, we have the following explicit form of the influence rate:
RI(H,O) = i ∑ |⃗a|Tr[[O,H]P⃗a]TrhOP†i+Trh[O,H]P†iTr[OP⃗a]. (17) d 2 n ⃗a ∈ V n ⃗a ⃗a
First, let us provide an upper bound on the influence rate.
Lemma 10. Given an n-qudit system with a Hamiltonian H and a linear operator O with ∥O∥2 = 1, we have
|RI(H,O)| ≤ 4n∥H∥∞ , (18) where ∥H∥∞ denotes the operator norm.
  Proof. Since |⃗a| 6 n, the Schwarz inequality yields
1 ∑ |⃗a||Tr[[O,H]P⃗a]| TrhOP†i ≤ n 1 ∑ |Tr[[O,H]P⃗a]| TrhOP†i
  d 2 n ⃗a ∈ V n ⃗a d 2 n ⃗a ∈ V n ⃗a = n∥[O,H]∥2 ∥O∥2 ≤ 2n∥H∥∞ ,
where the last inequality comes from the Hölder inequality and the fact that ∥O∥2 = 1. Similarly, we can prove that
1
d 2 n ⃗a ∈ V n ⃗a
∑ | ⃗a | T r h [ O , H ] P † i | T r [ O P ⃗a ] | ≤ 2 n ∥ H ∥ ∞ .
 
8 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE Hence, by the expression of influence rate in (17), we have
|RI(H,O)| ≤ 4n∥H∥∞ .
Let us provide an upper bound on the influence rate for the unitary generated
by a local Hamiltonian.
Theorem 11 (Small Incremental Influence). Given an n-qudit system with the Hamiltonian H acting on a k-qudit subsystem, and a linear operator O with unit norm ∥O∥2 = 1, one has
|RI(H,O)| ≤ 4k∥H∥∞ . (19) Proof. Since H acts on a k-qudit subsystem, there exists a subset S of size k such
thatH=HS⊗ISc.DefineO(1) on(Cd)Sc for⃗b∈VS by ⃗b
O(1)= 1 TrS[OP⃗]. (20) ⃗b dn−k b
Also define O(2) on (Cd)S for any⃗c ∈VSc as ⃗c
O(2)= 1 TrSc[OP⃗c]. (21) ⃗c dn−k
Notethat∑⃗ S O(1) 2 =∑ Sc O(2) 2 =1. DefiningA⃗ =O(1)/ O(1) and
B⃗c = O(2)/ O(2)
⃗c ⃗c 2
, we get that I[O] can be written as
b∈V ⃗b 2 ⃗c∈V ⃗c 2
b ⃗c ⃗c 2
(2) 2
(1) 2
I[B⃗c]+ ∑ O ⃗c ∈ V S b ∈ V S
I[O]= ∑ O

  Hence,
and so
† (2) 2 I[UtOUt ]= ∑ O
† (1) 2 I[UtB⃗cUt ]+ ∑ O
I[A⃗]. c ⃗c 2 ⃗ ⃗b 2 b
c ⃗c 2 ⃗c ∈ V S
I[A⃗], ⃗b 2 b
⃗b ∈ V S RI(HS,B⃗c).
RI(H,O)= ∑ O
(2) 2 ⃗c∈VSc ⃗c 2
Since both HS and B⃗c for any ⃗c ∈ V Sc act on a k-qudit subsystem, we have RI(HS,B⃗c)≤4k∥HS∥∞
by Lemma 10. Therefore, we obtain
as claimed.
|RI(H,O)| ≤ ∑ O(2) 2 |RI(HS,B⃗c)| ≤ 4k∥HS∥
,
⃗c ∈ V S c
⃗c2 ∞


COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 9
Here, we use circuit sensitivity to quantify the average sensitivity of a quantum circuit. In classical Boolean circuits, the average sensitivity of the circuit plays an important role in lower bounding the complexity of a circuit [23–25]. Hence, a natural question is: what is the connection between circuit sensitivity and circuit complexity for quantum circuits? Here, we use the circuit cost defined in [6] to quantify the complexity of quantum circuits. Our next result establishes a con- nection between the circuit sensitivity and the circuit cost of a quantum circuit.
Theorem 12 (Circuit Sensitivity Lower Bounds Circuit Cost). The circuit cost of a quantum circuit U ∈ SU(dn) is lower bounded by the circuit sensitivity as follows
Cost(U)≥ 1CiS[U]. (22) 8
Proof. The proof follows the same idea as that in [20, 65]. First, let us take a Trotter decomposition of U such that for arbitrarily small ε > 0,
∥U−VN∥∞ ≤ε, where VN is defined as follows
   and
N
VN := ∏Wt,
t=1
imt!
Wt := exp −N∑rj N hj . j=1
W = limW(l), t l→∞t
W(l) := W1/l ···W1/ll , t t,1 t,l
Wt,j := exp−irjthj. NN
  LetusdefineO =WO W† withO =O.ThenbyapplyingW,wehave ttt−1t 0 t
I[O]−I(O ) = IWO W†−I(O ) t t−1 tt−1t t−1
= limIW(l)O W(l)−I(O ) l→∞ t t−1t t−1
lm8t ≤ N∑l rj N
j=1 8mt
= N∑rj N , j=1
     
10 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE where the inequality above follows from Theorem 11 for k = 2. Taking the sum-
mation over all t, we have
† 8Nmt I(UOU)−I(O)≤N∑∑rj N .
  t=1 j=1 Since the circuit cost can be expressed as
Nmt Cost(U)= lim ∑∑ rj
,
 we have
which completes the proof of the theorem.

N→∞t=1 j=1 N I(UOU†)−I(O)≤8Cost(U),
2.3. Stable unitaries. Here we characterize quantum circuits with zero circuit sensitivity and provide a complete characterization of such unitaries.
Definition 13. An n-qudit unitary (or gate or circuit) U is stable if CiS[U ] = 0. Here, to characterize the stable unitaries, we need to consider weight-1 Pauli
operators, i.e. P⃗a with |⃗a| = 1.
Proposition 14. The circuit sensitivity satisfies the following three properties:
(1) An n-qudit unitary U is stable if and only if for any weight-1 Pauli oper- ator O, both U OU † and U †OU can be written as a linear combination of weight-1 Pauli operators.
(2) CiS[V2UV1] = CiS[U] for any unitary V1 and any stable unitary V2.
(3) CiS is subadditive under multiplication and tensorization:
CiS[UV] ≤ CiS[U]+CiS[V], CiS[U ⊗V] ≤ CiS[U]+CiS[V]. (23) Proof.
(1) If CiS[U ] = 0, for any weight-1 Pauli operator O, I[UOU†] = I[O] = 1.
Hence, UOU† can be written as a linear combination of weight-1 Pauli opera- tors. Similarly, U†OU can be written as a linear combination of weight-1 Pauli operators.
On the other hand, if it holds that for any weight-1 Pauli operator O, both UOU† and U†OU can be written as a linear combination of weight-1 Pauli op- erators, then UP⃗aU† and UP⃗aU† can be written as a linear combination of Pauli operators with weights less than |⃗a|. Hence, we have
Tr hP†U P⃗aU †i ̸= 0 only if |⃗a| = |⃗b|. (24) ⃗b
Let us define the transition matrix TU as follows
TU[⃗b,⃗a]= 1 TrhP†UP⃗aU†i, dn ⃗b
 
COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 11
forany⃗a,⃗b∈Vn. ItiseasytoseethatTU isaunitarymatrix. Here,duetothe condition (24), the unitary matrix can be decomposed as
n
TU =MT(k), U
k=0
where T(k) is a  n(d2 −1)k × n(d2 −1)k unitary matrix for any 0 ≤ k ≤ n,
Ukk
defined by T(k)[⃗b,⃗a] = 1 TrhP†UP U†i for any⃗a,⃗b with |⃗a| = |⃗b| = k. Hence,
and therefore,
U dn ⃗b⃗a
h† †i |⃗b|⃗
 ∑U ⃗a ⃗a:|⃗a|=|⃗b|
∑ PUOU†[⃗b]= ∑ PO[⃗b], ⃗b:|⃗b|=k ⃗b:|⃗b|=k
Tr PUOU = T [b,⃗a]Tr[OP],
⃗b
for any 0 ≤ k ≤ n. This implies that I[UOU†] = I[O]. Similarly, ∑ PU†OU[⃗b]= ∑ PO[⃗b],
⃗b:|⃗b|=k ⃗b:|⃗b|=k
and I[U†OU] = I[O]. Therefore, CiS[U] = 0.
(2) This statement follows directly from the definition.
(3) Subadditivity under multiplication comes directly from the triangle inequal- ity:
CiS[UV] ≤ max I[UVOV†U†]−I[VOV†] + max I[VOV†]−I[O] . O:∥O∥2 =1 O:∥O∥2 =1
Hence, to prove the subadditivity under tensorization, we only need to prove that CiS[U ⊗I] ≤ CiS[U]. Let us assume that U acts only on the k-qudit subsystem
S with k ≤ n. Similarly to the proof of Theorem 11, let us define O(1) on (Cd )Sc ⃗b
for any⃗b ∈ VS as (20) and, A⃗ = O(1)/ O(1) . Define O(2) on (Cd)S for any b⃗c⃗c2 ⃗c
⃗c∈VSc as(21)andB⃗c=O(2)/ O(2) ,soI[O]canbewrittenas ⃗c ⃗c 2
22 12 I[O]= O 2I[B ]+ Ob 2I[Ab].
⃗c ∈ V S c
Similarly, I[U ⊗IOU† ⊗I] can be written as
∑ ⃗c ⃗c ∑ ⃗ ⃗
† (2)2 † (1)2 I[U⊗IOU ⊗I]= ∑ O I[UB⃗cU ]+ ∑ O I[A⃗].
⃗b ∈ V S
c⃗c2 ⃗⃗b2b ⃗c ∈ V S b ∈ V S

12 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE Hence
|I[U ⊗IOU† ⊗I]−I[O]| ≤ ∑ O(2) 2 |I[UB⃗cU†]−I[B⃗c]| ⃗c∈VSc ⃗c 2
≤ CiS[U] ∑ O(2) 2 ⃗c∈VSc ⃗c 2
= CiS[U],
where we infer the second inequality from the definition of CiS. The last equality
comesfromthefactthat∑ Sc O(2) 2 =1. 
⃗c∈V ⃗c
We give two examples of stable unitaries. In fact, all stable unitaries can be
2 generated by these two types of unitaries.
(1) A Kronecker product of single-qudit unitaries, Nni=1 Ui. (2) Swap gates, i.e. the unitary mapping |ψ⟩|φ⟩ 7→ |φ⟩|ψ⟩.
Proposition 15. The set of stable unitaries is generated by the single-qudit uni- taries and the swap unitaries.
Proof. Given an n-qudit stable unitary U, let us consider its action on X1, where Xi denotes the Pauli operator X acting on the i-th qudit. Since U has zero circuit sensitivity, we have
UX1U†=∑αiQXi +∑βiQYi +∑γiQZi, i∈A i∈B i∈C
where QX is written as QX = ∑d−1 ci jX j with at least one coefficient ci j ̸= 0, i ij=1i
and A is the set of all indices i such that αi ̸= 0. The quantities QZi and C are similarlydefined.Moreover,QYisdefinedasQY=∑d−1 cijkXjZkwithatleast
i i j,k=1 ii
one coefficient ci jk ̸= 0, and B is the set of all indices i for which βi ̸= 0. Since
(UX1U†)2 = I, we have |A| ≤ 1, |B| ≤ 1 and |C| ≤ 1. The first inequality holds because if |A| ≥ 2, then there exists two indices i ̸= j such that (UX1U†)2 must containsometermQXi ⊗QXj,whichcontradictswiththefactthat(UX1U†)2=I.
Hence, we can simplify U X1U † as
U X 1 U † = α i Q Xi + β j Q Yj + γ k Q Zk .
Since (UX1U†)2 = I, we have i = j = k. This holds because if j ̸= i, then (UX1U†)2mustcontainthetermQXi ⊗QYj.Hence,wehave
Similarly, we have
U X 1 U † = α i Q Xi + β i Q Yi + γ i Q Zi . U Z 1 U † = α j Q Xj + β i Q Yj + γ i Q Zj .
If i ̸= j, then [UX1U†,UZ1U†] = 0, that is, [X1,Z1] = 0, which is impossible. Therefore i = j, i.e., there exists a local unitary V such that for any d × d matrix A , U A 1 ⊗ I n − 1 U † = A ′i ⊗ I n − 1 = V A i V † I n − 1 . H e n c e
V 1† S W A P 1 i U = I 1 ⊗ V 2 ,

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 13
where SWAP1i is the swap unitary between 1 and i, and V2 has zero circuit sen- sitivity on n − 1 qudits. By repeating the above process, we get that U can be generated by the local unitaries and swap unitaries. 
Stable unitaries also preserve multipartite entanglement, where the entangle- ment is quantified by the average Rényi-2 entanglement entropy:
S ̄(2)(ρ) = ES(2)(ρA),
where E := 1 ∑ denotes the expectation over subsets A ⊂ [n]; S(2)(ρ ) =
2 2n A⊂[n] A −logTrρA denotes the Rényi-2 entanglement entropy; and ρA denotes the re- duced state of ρ on the subset A.
 Corollary 16. Stable unitaries cannot increase the entanglement measure S ̄(2).
Proof. It is easy to verify that both local unitaries and swap unitaries will not change this average entanglement Rényi-2 entropy, so the corollary follows from the proposition. 
This shows that sensitivity is necessary for a quantum computational advan- tage.
Corollary 17. Given an n-qudit product state Nni=1 ρi as input, a stable quantum circuit U , and a single-qudit measurement set { N , I − N }, the outcome probabil- ity can be classically simulated in poly(n,d) time.
Proof. Since the stable quantum circuit can be generated by local unitaries and swap gates, such quantum circuits with product input states and local measure- ments can be simulated efficiently on a classical computer. 
2.4. Matchgates are Gaussian stable gates. In this section, we define variants of influence and circuit sensitivity, called Gaussian influence and Gaussian circuit sensitivity and show that matchgates have vanishing circuit sensitivity. We will show that Gaussian circuit sensitivity is necessary for a quantum computational advantage and that it provides a good measure to quantify the non-Gaussianity of quantum circuits. Let us consider the influence based on the generators of a Clifford algebra for an n-qubit system. First, we introduce 2n Hermitian operators γi which satisfy the Clifford algebra relations
{γi,γj } = 2δi,jI, ∀i, j = 1,...,2n. (25) Any linear operator can be expressed as a polynomial of degree at most 2n as
follows
where γS = ∏i∈S γi. Then
O= ∑OSγS, (26) S⊂[2n]
E Trh(γS)†Oi2=∥O∥2. (27) S∼U

14 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE Here the ES∼U denotes the expectation taken over all S ⊂ [2n] with respect to
theuniformdistribution,thatis,E = 1 ∑ .Hence,∥O∥ =1leadstoa S∼U 22n S⊂[2n] 2
probability distribution over S, which is defined as follows
PG[S] = 1 Tr h(γS)†Oi 2 , ∀S ⊂ [2n]. (28)
22n
Matchgates are an important family of tractable circuits, first proposed by Valiant in the context of counting problems [30]. Later, they were generalized to free fermionic quantum circuits, which are generated by a quadratic Hamiltonian intermsofCliffordgenerators{γi},i.e.,H=i∑i,jhijγiγj [34]. Oneimportant fact concerning matchgates (which are also called Gaussian gates) is that for each generator γi, UγiU† and U†γiU can always be written as linear combinations of γi [34].
We provide a new interpretation of matchgates via sensitivity, showing that they are the only unitaries which cannot change the influence. To obtain this result, define the influence with respect to the generators of the Clifford algebra {γi}i; we call this the Gaussian influence, to distinguish it from the previous definition.
Definition 18 (Gaussian Influence). Given a linear operator O, the local influ- ence at the j-th qudit is
I [O]= P [S], (29) Gj ∑OG
S: j∈S
and the total influence is the sum of all the local influences,
I [O]= I [O]= |S|P [S]. (30) G ∑Gj ∑OG
  O
j∈[2n]
Consider the Markov semigroup Pt introduced by Carlen and Lieb in [29],
S⊂[2n]
Pt(γS)=e−t|S|γS . (31)
Given an operator O, we have
∂ ∥Pt(O)∥2 = − ∑ |S|PG[S] = −IG[O]. (32)
 ∂t 2 t=0 O S⊂[2n]
Remark 1. There is no obvious relationship between the Pauli weight and the Gaussian weight. In particular, there exist operators whose Pauli weight is 1 and Gaussian weight is n, and also operators whose Gaussian weight is 1 and Pauli weight is n. Consequently, there is no obvious relationship between the total influence I and the total Gaussian influence IG.
Here, let us define the circuit sensitivity of a unitary with respect to IG. Definition 19. Given a unitary U, let us define the Gaussian circuit sensitivity
CiSG as the change of influence caused by the unitary evolution,
CiSG[U] = max IG[UOU†]−IG[O] . (33)
O:∥O∥2 =1
We say that U is Gaussian stable if CiSG[U] = 0.

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 15
Theorem 20. The Gaussian circuit sensitivity of an n-qudit unitary U satisfies the following three properties:
(1) The unitary U is Gaussian stable if and only if U is a matchagte. (2) CiSG[V2UV1] = CiSG[U] for any unitary V1 and matchgate V2. (3) CiSG is subadditive under multiplication,
CiSG[UV] ≤ CiSG[U]+CiSG[V]. (34)
Proof.
(1) On one hand, if CiSG[U] = 0, then for any generator γi,
IG[UγiU†] = IG[γi] = 1.
Hence, UγiU† can be written as ∑j cjγj. Similarly, U†γiU can be written as a
linear combination of { γ j } j . Ontheotherhand,ifforanygeneratorγi,bothUγiU† andU†γiU canbewritten
as a linear combination of {γj }j, then UγSU† and UγSU† can be written as a linear combination of {γS′ : S′ ⊂ [2n], | S′| ≤ |S|}. Hence, we have
Trh(γS′)†UγSU†i ̸= 0 only if |S′| = |S|. (35) Let us define the transition matrix TU as follows
TU[S1,S2]= 1 Trh(γS1)†UγS2U†i, 2n
for any S1,S2 ⊂ [2n]. It is easy to see that TU is a unitary matrix. Here, due to condition (35), the unitary matrix can be decomposed as
 2n
TU =MT(k),
where T (k) is a  2n ×  2n unitary matrix for any 0 ≤ k ≤ 2n, and defined as Ukk
T(k)[S ,S ]= 1 Tr(γS1)†UγS2U†foranyS ,S with|S |=|S |=k.Hence, U 1 2 2n 1 2 1 2
Tr (γ 1) UOU = and therefore,
T [S ,S ]Tr (γ 2) O , ∑U 12
∑
S2 :|S2 |=|S1 | GG
P UOU
[S ] = P [S ], †1 ∑O1
S1 :|S1 |=k
for any 0 ≤ k ≤ n. This implies that IG[UOU†] = IG[O]. Similarly,
P [S ] = P [S ], †1∑O1
S1 :|S1 |=k
∑
S1 :|S1 |=k
S1 :|S1 |=k GG
U OU
U k=0
 hS† †i (|S1|) hS†i
and we have IG[U†OU] = IG[O]. Therefore, CiSG[U] = 0.
(2) It follows directly from the definition.
(3) It follows directly from the triangle inequality. 

16 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
Since matchgates can be simulated efficiently on a classical computer, the Gaussian stable gates cannot yield a quantum advantage. From this we infer that Gaussian sensitivity is necessary for a quantum computational advantage. Since matchgates are sometimes called Gaussian operations and Gaussian circuit sen- sitivity can be used as a measure to quantify the non-Gaussian nature of quantum circuits, the Gaussian circuit sensitivity shows how "non-matchgate" a circuit is.
The set of stable gates (i.e., CiS = 0) and Gaussian stable gates (i.e., CiSG = 0) are quite different. For example, for an n-qubit system, the SWAP gates are stable but not Gaussian stable; on the other hand, the nearest neighbor (n.n.) G(Z,X) gate is Gaussian stable, but not stable. Here the gate G(Z,X) is defined as
stable unitaries
SWAP
Gaussian stable unitaries
n.n. G(Z,X)
1000 G(Z,X)=0 0 1 0 .
 0 1 0 0  0 0 0 −1
Complementing this, we remark that a single-qubit unitary acting on the first qubit U1 lies in the overlap of the two sets. We illustrate this in Figure 1.
  U1
FIGURE 1. A Venn diagram illustrating the overlap between the stable gate set and the Gaussian stable gate set on n-qubit systems, as explained in the text.
Remark 2. Here we consider the sensitivity of quantum circuits with respect to noise, where we define the stable gates (or circuits) as the gates with zero sensitivity. The circuit sensitivity (or influence) may be used to quantify the classical simulation time, a question we plan to study in the future.
In classical computation, algorithmic stability is one of the fundamental prop- erties of a classical algorithm, and it plays an important role in computational learning theory. For example, it gives insight into the differential privacy of randomized algorithms [66, 67], into the generalization error of learning algo- rithms [68, 69], and so on. This implies that algorithmic stability is useful to understand learning. Hence, one defines quantum algorithmic stability via influ- ence (or circuit sensitivity) for quantum algorithms or circuits as a generalization of the classical theory. One can then study its application in quantum differential

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 17
privacy [70, 71] and in understanding the generalization error of quantum ma- chine learning [72–78]. Besides, the stable gates (or circuits) can be efficiently simulated on a classical computer, which shows that stability may not imply a quantum speedup.
In summary, there appears to be a trade-off between quantum computational speedup and the capability of generalization in quantum machine learning.
2.5. Quantifying scrambling by influence on average case. Here we clarify the relationship between influence and scrambling. Information scrambling mea- sures the delocalization of quantum information by chaotic evolution. Scram- bling prevents one from determining the initial conditions that precede chaotic evolution through the use of local measurements. One well-known measure of scrambling is the out-of-time-ordered commutator (OTOC). This is defined as the Hilbert-Schmidt norm of the commutator between two initially commuting local Pauli strings after one operator evolves under the action of a unitary. Scrambling refers to the speed of growth of the OTOC. Mathematically, the OTOC is defined by
C(t) = 1 ∥[OD(t),OA]∥2 = 1−⟨OD(t)OAOD(t)OA⟩ , (36) 2
where OD(t) := UtODUt†, the expectation value ⟨·⟩ is taken with respect to the n-qubit maximally mixed state I/dn, and A,D denote two disjoint subregions of the n-qudit system. For simplicity, we take the local dimension to be d = 2, i.e., the systems we consider are qubit systems.
If we restrict the regions A and D to be 1-qubit systems, then the average OTOC over all possible positions for A can be expressed in terms of the influence of OD(t). Without loss of generality, let us assume that the region D is taken to be the n-th qubit.
Proposition 21 (Average OTOC-Influence Relation). If the region D is the n-th qubit, then
 d2 1n−1 EAEOA⟨OD(t)OAOD(t)OA⟩=1−d2−1n−1 ∑Ij[OD(t)],
j=1
  where EA denotes the average over all positions j ∈ [n − 1] such that OA initially commuteswithOD,andEOA denotestheaverageoveralllocalnon-identityPauli operators on position A.
Proof. Since ⟨OD(t)OAOD(t)OA⟩ can be written as the linear combination of the terms⟨P⃗aPcjP⃗bPcj⟩with⃗a,⃗b∈Vn andPcj beingthelocalnon-identityPaulioper- atoronthe j-thqubit,wefirstconsidertheaverageof⟨P⃗aPcjP⃗bPcj⟩withPcj taking
(37)

18 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE on all non-identity Pauli operators uniformly,
E P ⟨ P⃗a P j P⃗ P j ⟩ = j
∑ ⟨ P⃗a P j P⃗ P j ⟩ j=(s,t)∈V\(0,0)
1
b d2−1 b
 = δ 1 (d2δ −1) ⃗a ,⃗b d 2 − 1 a j , 0
= δ⃗1−|aj| d2 . ⃗a,b d2−1
Hence,anyO (t)canbewrittenasO (t)=∑ 1 Tr[PO (t)]P, D D ⃗adn ⃗aD ⃗a
   1n−1  d2 EAEOA⟨OD(t)OAOD(t)OA⟩ = n−1 ∑ ∑ 1−|aj|d2 −1
POD[⃗a] d2 1 "n−1 #
  j = 1 ⃗a ∈ V n
= 1−d2−1·n−1 ∑n ∑|aj| POD[⃗a]
  ⃗a∈V j=1 d2 1n−1
= 1−d2−1·n−1∑∑n|aj|POD[⃗a] j = 1 ⃗a ∈ V
d2 1n−1
= 1−d2−1n−1 ∑Ij[OD(t)],
j=1
where Ij is defined as Ij[O] = ∑aj̸=0 PO[⃗a]. 
    Proposition 21 ensures that the average OTOC tends to d2 1n−1 d21
1−d2−1n−1 ∑Ij[OD(t)]→1−d2−1nI[OD(t)]asn→∞. j=1
    This provides the relations between scrambling and the total influence. Aside from the OTOC, higher-order OTOCs, such as the 8-point correlator, can also be related to the total influence on average (See Appendix A).
3. QUANTUM FOURIER ENTROPY AND INFLUENCE
Here, we define the quantum Fourier entropy H[O] and show its relationship with the influence I[O]. We shall show that the quantum Fourier entropy can be used as a measure of magic in quantum circuits, which we call the “magic entropy”. In addition, we use results on quantum Fourier entropy and influence to obtain the relations between magic and sensitivity (or Gaussian sensitivity).
3.1. Quantum Fourier entropy-influence relation and conjecture. Definition 22 (Quantum Fourier Entropy and Min-entropy). Given a linear
n-qudit operator O with ∥O∥2 = 1, the quantum Fourier entropy H[O] is
H[O]=H[PO]=− ∑PO[⃗a]logPO[⃗a], (38) ⃗a ∈ V n

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 19 with {PO[⃗a]} being the probability distribution defined in (6). The quantum
Fourier min-entropy H∞[O] is
H∞[O]=H∞[PO]=minlog 1 . (39)
 ⃗a ∈ V n P O [ ⃗a ]
One can also define the quantum Fourier Rényi entropy as
Hα[O]=Hα[PO]= 1 log ∑Pα[⃗a]!. (40)
In the study of classical Boolean functions, Friedgut and Kalai proposed the now well-known Fourier entropy-influence conjecture [52]. Another well-known, but weak, conjecture is the Fourier min-entropy-influence conjecture. Appen- dix B provides a brief introduction to the Fourier entropy-influence conjecture for Boolean functions.
Theorem 23 (Weak QFEI). For any linear operator O on an n-qudit system with ∥O∥2 = 1, we have
H[O] ≤ c[logn+logd]I[O]+h[PO[⃗0]], (41) where h(x) := −xlogx−(1−x)log(1−x) is the binary entropy and c is a uni-
versal constant. Here, c can be taken to be 2.
Proof. Let us define a new probability distribution {Wk[O]}k on the set [n] as
 follows
Therefore, the total influence I[O] can be rewritten as I[O]= ∑|⃗a|PO[⃗a]=∑kWk[O].
⃗a ∈ V n k
Hence, the quantum Fourier entropy can be written as
1 H[O] = ∑PO[⃗a]logP[⃗a]
1−α
O ⃗a
W k [ O ] = ∑ P O [ ⃗a ] . ⃗a ∈ V n : | ⃗a | = k
 nO
W | ⃗a | [ O ] 1
⃗a∈V
= ∑PO[⃗a]logP[⃗a]+logW[O]
!
  ⃗a ∈ V n O | ⃗a | W | ⃗a | [ O ]
1
= ∑PO[⃗a]logP[⃗a]+∑PO[⃗a]logW[O]
  ⃗a ∈ V n O ⃗a ∈ V n | ⃗a |
P O [ ⃗a ] W k [ O ]
k ⃗a:|⃗a|=k k
1
= ∑Wk[O] ∑ Wk[O]log PO[⃗a] +∑Wk[O]logWk[O].
   
20 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE Note that if Wk[O] ̸= 0, then PO[⃗a] is a probability distribution on the set Sk =
 Wk[O] {⃗a∈Vn :|⃗a|=k}. Hence
k  ∑Wk[O]k(logn+log(d2 −1))
P O [ ⃗a ] W k [ O ]   n  2
∑ W k [ O ] l o g P O [ ⃗a ] ≤ l o g | S k | ≤ l o g k ( d − 1 )
  ⃗a:|⃗a|=k Therefore, we have
k(logn+log(d2 −1)). k
≤ ∑Wk[O] ∑ PO[⃗a] logWk[O] ≤
  k ⃗a:|⃗a|=k Wk[O] PO[⃗a]
=
[logn+log(d2 −1)]I[O]. Next,letusprovethat∑kWk[O]log 1 ≤I[O]+h(PO[⃗0]).First,ifTr[O]=0,
 Wk[O]
then H[O] ≤ I[O]. This comes from the positivity of the relative entropy between
the probability distributions W⃗ = {Wk[O]}k and ⃗p = { pk }k, with pk = 2−k for 1≤k≤nand p0 =2−n,whichcanbeexpressedas
⃗ n Wk[O]
D(W∥⃗p) = ∑Wk[O]log pk[O] = ∑kWk[O]+∑Wk[O]logWk[O] ≥ 0.
k=0 k k If Tr [O] ̸= 0, let us us define a new operator
O ′ = 1 ∑ O ⃗a P ⃗a . 1 − W 0 [ O ] ⃗a ̸ = 0
Then for this new operator O′, we have H[O′] ≤ I[O′],
  and Hence,
I[O]=(1−W0[O])I[O′].
1 ∑Wk[O]logWk[O]
 k
= W0[O]logW0[O]+∑Wk[O]logWk[O]
11 k≥1
1′1
= W0[O]logW0[O]+∑(1−W0[O])Wk[O]log(1−W0[O])Wk[O′]
    k≥1
= W0[O]log 1 +(1−W0[O])log 1
  W0[O] (1−W0[O]) +(1−W0[O]) ∑Wk[O′]log 1
 ≤ h(W0[O])+(1−W0)I[O′] = I [ O ] + h ( P O [ ⃗0 ] ) ,
k≥1 Wk[O′]

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 21 wherehdenotesthebinaryentropyh(x)=−xlogx−(1−x)log(1−x). This
completes the proof of the theorem.  Now, let us consider the quantum Fourier entropy-influence conjecture on qubit
systems, which improves upon Theorem 23.
Conjecture 24 (Quantum Fourier Entropy-Influence Conjecture). Given a
Hermitian operator O on n-qubit systems with O2 = I,
H[O] ≤ cI[O], (42)
where the constant c is independent of n.
Proposition 25 (QFEI Implies FEI). If QFEI is true for Hermitian operators O
on n-qubit system with O2 = I, then FEI is also true.
Proof. Consider any function f : {−1,1}n → {−1,1} with the corresponding
Fourier expansion f (x) = ∑S⊂[n] fˆ(S)xS. Let us define the following observable Of = ∑ fˆ(S)XS.
S⊂[n]
where XS := ∏i∈S Xi and Xi is the Pauli X operator on the i-th qubit. Of is a
Hermitian operator with O2f = I. Note that
Of |x⟩= f(x)|x⟩, ∀x∈{−1,1}n,
where|±1⟩= 1 (|0⟩+|1⟩)and|0⟩,|1⟩aretheeigenstatesofthePauliZoperator. √2
Hence ⟨Of ,P⃗a⟩ = fˆ(S) when P⃗a = XS, and ⟨Of ,P⃗a⟩ = 0 otherwise. That is, H[f] = H[Of],
I[f] = I[Of].
This completes the proof of the proposition. 
Similarly, QFMEI is a quantum generalization of FMEI.
Proposition 26 (QFMEI Implies FMEI). If QFMEI is true for all quantum
Boolean functions, FMEI is also true.
Proof. The proof follows the same lines as the proof of Proposition 25. 
3.2. Magicentropy-circuitsensitivityrelation. Magicisanimportantresource in quantum computation, as a quantum circuit without magic provides no quan- tum advantage. The Gottesman-Knill theorem states that Clifford unitaries with stabilizer states and Pauli measurements can be efficiently simulated on a clas- sical computer [37, 79]. Here, a Clifford unitary is defined as a unitary which maps a Pauli operator to a Pauli operator. Since any Pauli operator is generated by the product of weight-1 Pauli operators, the Clifford unitaries are precisely those unitaries which map any weight-1 Pauli operator to a Pauli operator. For a non-Clifford unitary, an important task is to quantify the amount of magic in the unitary. Here, we introduce a new concept, which we call the magic entropy.
  
22 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE Definition 27 (Magic Entropy). Given a unitary U, the magic entropy M[U] is
M[U] := max H[UOU†]. (43) O: weight-1 Pauli
Since the quantum Fourier entropy of any weight-1 Pauli is always 0, the magic entropy can also be written as follows
M[U]= max (H[UOU†]−H[O]), (44) O: weight-1 Pauli
which also quantifies the change of quantum Fourier entropy on weight-1 Pauli operators.
Proposition 28. The magic entropy M[U] satisfies the following three properties:
(1) Faithfulness: M[U] ≥ 0, and M[U] = 0 if and only if U is a Clifford unitary.
(2) Invariance under multiplication by Clifford unitaries: M[VU] = M[U] for any Clifford unitary V .
(3) Maximization under tensorization: M[U1 ⊗U2] = max{M[U1],M[U2]} for any unitaries U1 and U2.
Proof. These properties follow directly from the definition of magic entropy. 
Example 1. Let us consider a widely-used single-qubit non-Clifford T gate,
whichisdefinedasT = 1 0 . ThemagicentropyofT isM[T]=1. 0 eiπ/4
Based on the relations between quantum Fourier entropy and influence in §3.1, we can obtain the connection between magic entropy and circuit sensitivity.
Proposition 29 (Magic-Sensitivity Relation). Given an n-qudit unitary U , the magic entropy and circuit sensitivity satisfy the following relation:
M[U]≤c[logn+logd](CiS[U]+1). (45) Proof. Based on Theorem 23, we have
H[UOU†] ≤ c[logn+logd]I[UOU†]. Besides, as I[O] = 1 for a weight-1 Pauli operator O, we have
I[UOU†]≤CiS[U]+I[O]=CiS[U]+1.
Thus
for any weight-1 Pauli operator O. 
H[UOU†] ≤ c[logn+logd](CiS[U]+1),
Proposition 30. If the QFEI conjecture holds for an n-qubit system, then for any
n-qubit unitary U ,
M[U]≤c(CiS[U]+1). (46) Proof. The proof is similar to that for Proposition 29. 

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 23
Since the Gaussian influence IG has properties similar to the influence I, we can get the following connection between quantum Fourier entropy and Gaussian influence by a similar proof, which we call the weak Quantum Fourier entropy- Gaussian influence relation (QFEGI). Hence, it also implies the connection be- tween magic entropy and Gaussian circuit sensitivity for quantum circuits.
Theorem 31 (Weak QFEGI). For any linear operator O on an n-qubit system with ∥O∥2 = 1, we have
H[O] ≤ clog(2n)IG[O]+h[PO[⃗0]], (47) where h(x) := −xlogx−(1−x)log(1−x) is the binary entropy and c is a uni-
versal constant.
Proof. The proof is similar to that of Theorem 23, so we omit it here. 
Proposition 32 (Magic-Gaussian Sensitivity Relation). Given an n-qudit uni- tary U, the magic entropy and Gaussian circuit sensitivity satisfy the following relation
M[U]≤c(log2n)(CiSG[U]+1). (48) Proof. The proof is similar to that of Theorem 29, so we omit it here. 
4. MAGIC AND CIRCUIT COMPLEXITY
4.1. A lower bound on circuit cost from magic-influence relation. As the in- fluence of a unitary evolution can provide a lower bound on the circuit cost, the magic-influence relation directly implies a lower bound on the circuit cost by the amount of magic.
Proposition 33. The circuit cost of a unitary U ∈ SU(dn) satisfies the following lower bound given by the magic entropy
Cost(U)+1≥ 1 M[U]. (49) cd logn
Proof. This is because
M[U ] ≤ cd log(n)(CiS[U ] + 1) ≤ cd log(n)(Cost(U ) + 1),
where the first inequality comes from Proposition 29, and the second inequality comes from Theorem 12. 
Proposition 34. If the QFEI conjecture holds for n-qubit systems, then the circuit cost of a unitary U ∈ SU(2n) satisfies the following bound
Cost(U)+1≥ 1M[U]. (50) c
Proof. This is because
M[U] ≤ c(CiS[U]+1) ≤ c[Cost(U)+1],
where the first inequality comes from Proposition 30, and the second inequality comes from Theorem 12.

  
24 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
4.2. A lower bound on circuit cost by magic power. In subsection §4.1, we obtain a lower bound on the circuit cost based on the magic-influence relation. This lower bound has a logn factor, which can be removed under the quantum Fourier entropy-influence conjecture. In this subsection, our goal is to get rid of the logn factor without the conjecture. First, let us introduce another concept called magic power, which is a generalization of magic entropy.
Definition 35 (Magic Power). Given a unitary U , the magic power M [U ] is the maximal magic generated by U ,
M[U]= max H[UOU†]−H[O] . (51) O:∥O∥2 =1
It is easy to see that the magic power satisfies M[U] ≥ M[U]. Let us first discuss some properties of the magic power.
Lemma 36. The magic power equals
M[U]= max H[UOU†]−H[O] , (52)
O:∥O∥2 =1,Tr [O]=0
that is, the maximization is taken over all traceless operators with ∥O∥2 = 1.
Proof. Let us define a new operator O′:
′ 1Tr[O]
O=q O− dn I . 1 − P O [ ⃗0 ]
   If Tr[O] ̸= 0, then O′ satisfies the condition Tr[O′] = 0 and ∥O′∥2 = 1. Since P †[⃗0] = P [⃗0], H[UOU†] and H[O] can be rewritten as
H [ O ] = h h P O [ ⃗ 0 ] i + ( 1 − P O [ O⃗ ] ) M [ O ′ ] ,
H [ U O U † ] = h h P O [ ⃗ 0 ] i + ( 1 − P O [ O⃗ ] ) M [ U O ′ U † ] .
Hence we have H[UOU†]−H[O]=(1−PO[⃗0])(H[UO′U†]−H[O′]).
Therefore, the maximization is obtained from traceless operators.  Proposition 37. The magic power M [U ] satisfies the following three properties;
(1) Magic power is faithful: M [U ] ≥ 0, and M [U ] = 0 if and only if U is a Clifford unitary.
(2) Magic power is invariant under multiplication by Cliffords: M [V2UV1] = M [U] for any unitary V1 and Clifford unitary V2.
(3) Magic power is subadditive under multiplication and tensorization: M[UV] ≤ M[U]+M[V], M[U ⊗V] ≤ M[U]+M[V]. (53)
Proof.
(1) M [U] ≥ 0 comes directly from the definition of M [U]. If M [U] = 0, it implies that H[UP⃗aU†] = 0 for any Pauli operator P⃗a, that is UP⃗aU† is a Pauli op- erator. Hence the unitary U is a Clifford unitary. If U is a Clifford unitary—i.e. if
U O U O⃗

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 25
U always maps Pauli operators to Pauli operators—then the probability distri- bution{PUOU†[⃗a]}isequivalentto{PO[⃗a]}uptosomepermutation. Hence H[UOU†] = H[O].
(2) This follows directly from the definition of M [U ].
(3) Subadditivity under multiplication comes directly from the triangle inequality, that is
M[UV] ≤ max H[UVOV†U†]−H[VOV†] + max H[VOV†]−H[O] O:∥O∥2 =1 O:∥O∥2 =1
= M[U]+M[V].
Hence, to prove subadditivity under tensorization, we only need to prove that M [U ⊗ I] ≤ M [U]. Let us assume that U acts on only a k-qudit subsystem S with k ≤ n. Let us define O⃗c on (Cd)S for any⃗c ∈VSc as follows
O⃗c= 1 TrSc[OP⃗c], (54) dn−k
and it is easy to verify that ∑⃗c∈VSc ∥O⃗c∥2 = 1. Defining B⃗c = O⃗c/∥O⃗c∥2, we get that H[O] can be written as
H[O]=∑∥O⃗c∥2H[B⃗c]−∑∥O⃗c∥2log∥O⃗c∥2. ⃗c ⃗c
Similarly,
H[U ⊗IOU† ⊗I] = −∑⃗c ∥O⃗c∥2 H[UB⃗cU†]−∑⃗c ∥O⃗c∥2 log∥O⃗c∥2 .
Hence
|H[U⊗IOU†⊗I]−H[O]|≤∑⃗c ∥O⃗c∥2 H[UB⃗cU†]−H[B⃗c] ≤M[U].
Hence, we obtain the result. 
Example 2. By a simple calculation, the magic power of a T gate is M [T ] = 1. Moreover,forncopiestheT gate,namelyT⊗n,itsmagicpowerisM[T⊗n]=n, whereas its magic entropy M[T⊗n] = 1, which follows directly from the max- imization of magic entropy under tensorization. This example illustrates that magic power may be much larger than magic entropy for the same unitary.
We now introduce the magic rate, which can be used to quantify small incre- mental magic for a given unitary evolution.
Definition 38 (Magic Rate). Given an n-qudit Hermitian Hamiltonian H and a linear operator O with ∥O∥2 = 1, the magic rate of the unitary Ut = e−itH acting on O is
R (H,O)= dH[UOU†] . (55) M dt t t t=0
  
26 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
First, let us provide an analytic formula for the magic rate by a direct calcula- tion as follows,
RM(H,O) = i ∑ Tr[[O,H]P⃗a]TrhOP†ilogPO[⃗a] d 2 n ⃗a ∈ V n ⃗a
+ T r h [ O , H ] P † i T r [ O P ⃗a ] l o g P O [ ⃗a ]  . ⃗a
Lemma 39. Consider the function g(x) = x(logx)2 with x ∈ [0,1]. Then 0 ≤ g(x) ≤ g(e−2) = (2loge)2/e2 for x ∈ [0,1]. Moreover, g(x) is increasing on [0, e−2] and decreasing on [e−2, 1].
Proof. This lemma follows from elementary calculus. See Fig. 2 for a plot of the function g(x). 
 g(x) 1.1267
FIGURE 2. A plot of the function g(x) = x(logx)2 for x ∈ [0,1], where the logarithm is taken to be of base 2. The maximum value of g(x) is g(e−2) ≈ 1.1267, which occurs at x = e−2 ≈ 0.135. The function g(x) vanishes at both x = 0 and x = 1. In addition, it is increasing on [0,e−2] and decreasing on [e−2,1].
Lemma 40. Given an n-qudit Hamiltonian H and a linear operator O with
∥O∥2 = 1, we have
|RM(H,O)| ≤ 8dn ∥H∥∞ log(e)/e. (56)
Proof. The Schwarz inequality yields
1/2
≤ ∥ [ H , O ] ∥ 2 ∑ P O [ ⃗a ] l o g 2 P O [ ⃗a ]
⃗a ∈ V n
≤ ∥[H,O]∥2 (qd2ng(e−1))1/2
  g(x) = x(log x)2
  0 e−2 1 x
1
d 2 n ⃗a ∈ V n ⃗a
|Tr[[O,H]P ]| Tr OP logP [⃗a] ∑ ⃗a h †i! O
  ≤ 2dn ∥H∥∞ g(e−2),

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 27
where the second inequality come from the fact that g(x) ≤ g(e−2) and the last inequality comes from the Hölder inequality. Similarly,
1 ∑ |Trh[O,H]P†iTr[OP⃗a]logPO[⃗a]| ≤ 2dn ∥H∥∞ qg(e−2). (57) d 2 n ⃗a ∈ V n ⃗a
Therefore, we get the bound in (56). 
Theorem 41 (Small Incremental Magic). Given an n-qudit system with the Hamiltonian H acting on a k-qudit subsystem, and a linear operator O with
∥O∥2 = 1, one has
|RM(H,O)| ≤ 8dk ∥H∥∞ log(e)/e. (58)
Proof. Since H acts on a k-qudit subsystem, there exists a subset S of size k such thatH=HS⊗ISc.DefineO⃗c on(Cd)S for⃗c∈VSc by
O⃗c= 1 TrSc[OP⃗c]. dn−k
Note that ∑⃗c∈VSc ∥O⃗c∥2 = 1. Define B⃗c = O⃗c/∥O⃗c∥2. Then, H[O] can be written as
H [ U t O U t † ] = ∑ ∥ O ⃗c ∥ 2 2 H [ U t B ⃗c U t † ] − ∑ ∥ O ⃗c ∥ 2 2 l o g ∥ O ⃗c ∥ 2 2 . ⃗c ⃗c
Hence,
RM(O,H) = ∑∥O⃗c∥2 RM(B⃗c,HS). ⃗c
Then, by Lemma 40, we have |RM(B⃗c,HS)| ≤ 4dk ∥HS∥∞. Therefore, we have |RM(O,H)|≤∑∥O⃗c∥2|RM(B⃗c,HS)|≤4dk∥H∥∞. (59)
⃗c

In (58), the dependence on the local dimension d occurs as O(dk). In §5, the connection between coherence and circuit complexity is studied, where we show that the dependence on the local dimension is O(klogd). This suggests that a similar bound may also hold for magic.
Conjecture 42. Given an n-qudit system with the Hamiltonian H acting on a k-qudit subsystem,
?
|RM(H,O)| ≤ cklog(d)∥H∥∞ , (60)
where c is a constant independent of k,d, and n.
Theorem 43 (Magic power bounds the circuit cost). The circuit cost of a quan-
   tum circuit U ∈ SU(dn) is lower bounded by the magic power as follows Cost(U)≥ e M[U]. (61)
Proof. The proof is almost the same as that of Theorem 12, which we omit here. 
 8d2 log(e)

28 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
Corollary 44. If Conjecture 42 holds, then the circuit cost of a quantum circuit U ∈ SU(dn) is lower bounded by the magic power as follows
Cost(U)≥ c M[U]. (62) log d
Proof. The proof is almost the same as that of Theorem 43, which we omit here. 
5. COHERENCE AND CIRCUIT COMPLEXITY
First, let us recall the basic concepts in the resource theory of coherence. Given a fixed reference basis B = { | i⟩}i, any state which is diagonal in the reference basis is called an incoherent state. The set of all incoherent states is denoted as I . To quantify the coherence in a state, we need to define a coherence measure. Examples of such measures include the l1 norm coherence and relative entropy of coherence [60]. In this work, we focus on the relative entropy of coherence, which is defined as follows
Cr(ρ) = S(∆(ρ))−S(ρ), (63)
whereS(ρ):=−Tr[ρlogρ]isthevonNeumannentropyofρ and∆(·):=∑i⟨i|· |i⟩|i⟩⟨i| is the completely dephasing channel. This allows us to define the cohering power for a unitary evolution U as:
Cr(U) = max |Cr(UρU†)−Cr(ρ)|. (64) ρ∈D((Cd)⊗n)
where the maximization is taken over all density operators ρ ∈ D((Cd)⊗n).
Definition 45 (Rate of Coherence). Given an n-qudit Hamiltonian H and a quan- tum state ρ, the coherence rate RC(H,ρ) is the derivative of the coherence mea- sure with respect to time t at t = 0:
 RC(H,ρ):= dCre−itHρeitH
dt t=0
. (65) Lemma 46. Given a Hamiltonian H on an n-qudit system and an n-qudit quan-
 tum state ρ, the coherence rate RC(H,ρ) can be written
RC(H,ρ) = −iTr[[ρ,log∆(ρ)]H]. (66)
Proof. This comes from direct calculation.  Proposition 47. Given an n-qudit system with a Hamiltonian H and an n-qudit
quantum state ρ ∈ D((Cd)⊗n), the coherence rate satisfies the following bound |RC(H,ρ)| ≤ 4∥H∥∞ Dmax(ρ∥∆(ρ)), (67)
where Dmax is the maximal relative entropy defined as
Dmax(ρ∥σ) = logmin{λ : ρ ≤ λσ }. (68)
Proof. To prove this result, we need the following lemma.

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 29
Lemma 48. (Mariën et al. [65]) Given two positive operators A and B with A ≤ B and Tr [B] = 1, there exists a universal constant c such that
Tr [|[A, log B]|] ≤ 4 h( p), (69) where p = Tr [A], and h( p) = − p log p − (1 − p) log(1 − p). Here, c can be taken
to be 4 [80].
The proof of Proposition 47 is a corollary of the above lemma by taking A =
2−Dmax(ρ∥∆(ρ))ρ and B = ∆(ρ).  Theorem 49. Given an n-qudit system with the Hamiltonian H acting a k-qudit
subsystem and an n-qudit quantum state ρ ∈ D((Cd)⊗n), we have
|RC(H,ρ)| ≤ 4∥H∥∞ klog(d). (70)
Proof. Since H acts on a k-qudit subsystem, there exists a subset S ⊂ [n] with |S| = k such that H = HS ⊗ISc. Based on Lemma 46, we have
RC(H,ρ)=−i ∑ ⟨⃗z|[HS⊗ISc,ρ]|⃗z⟩logp(⃗z). ⃗z∈[d]n
Let us decompose |⃗z⟩ = |⃗x⟩|⃗y⟩, where⃗x ∈ [d]S and⃗y ∈ [d]Sc. Then we have RC(H,ρ)
= −i ∑ ⟨⃗x|⟨⃗y|[HS ⊗ISc,ρ]|⃗x⟩|⃗y⟩logTr[ρ|⃗x⟩⟨⃗x|⊗|⃗y⟩⟨⃗y|]. ⃗x ∈ [ d ] S , ⃗y ∈ [ d ] S c
Now, let us define a set of k-qudit states { ρ⃗y }⃗y as follows ρ⃗y := TrSc[ρ|⃗y⟩⟨⃗y|Sc],
p⃗y
for any ⃗y ∈ [d]Sc , where the probability p⃗y is defined as
p ⃗y = T r [ ρ | ⃗y ⟩ ⟨ ⃗y | S c ⊗ I S ] .
Note that ∑⃗y p⃗y = 1. Hence, RC(H,ρ) can be rewritten as
RC(H,ρ) = −i ∑ ∑ ⟨⃗x|[HS,ρ⃗y]|⃗x⟩p⃗ylog(Trρ⃗y|⃗x⟩⟨⃗x|p⃗y) ⃗x ∈ [ d ] S ⃗y ∈ [ d ] S c
= −i ∑ ∑ ⟨⃗x|[HS,ρ⃗y]|⃗x⟩p⃗ylogTrρ⃗y|⃗x⟩⟨⃗x| ⃗x ∈ [ d ] S ⃗y ∈ [ d ] S c
−i ∑ ∑ ⟨⃗x|[HS,ρ⃗y]|⃗x⟩p⃗ylogp⃗y. ⃗x ∈ [ d ] S ⃗y ∈ [ d ] S c
Since ∑⃗x∈[d]S ⟨⃗x|[HS,ρ⃗y]|⃗x⟩ = Tr[HS,ρ⃗y] = 0, we have
i ∑ ∑ ⟨⃗x| [HS, |ρ⃗y⟩⟨ρ⃗y|] |⃗x⟩ p⃗y log p⃗y = 0. ⃗x ∈ [ d ] S ⃗y ∈ [ d ] S c
 
30 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE Therefore,
RC(H,ρ) = −i ∑ ∑ ⟨⃗x|[HS,ρ⃗y]|⃗x⟩p⃗ylogTrρ⃗y|⃗x⟩⟨⃗x| ⃗x ∈ [ d ] S ⃗y ∈ [ d ] S c
= ∑ p⃗y−i ∑ ⟨⃗x|[HS,ρ⃗y]|⃗x⟩logTrρ⃗y|⃗x⟩⟨⃗x| ⃗y∈[d]Sc ⃗x∈[d]S
= ∑ p⃗yRC(H,ρ⃗y). ⃗y∈[d]Sc
By Proposition 47, we have
|RC(H,ρ)| ≤ 4 ∑ p⃗y ∥HS∥∞ Dmax(ρ⃗y∥∆(ρ⃗y)).
⃗y∈[d]Sc
Since ρ⃗y is a quantum state on a k-qudit system, Dmax(ρ⃗y∥∆(ρ⃗y)) ≤ klog(d).
Hence, we have
|RC(H,ρ)|≤4k∥HS∥∞log(d).
Theorem 50. [Cohering power lower bounds the circuit cost] The circuit cost of
a quantum circuit U ∈ SU(dn) is lower bounded by the cohering power as follows Cost(U) ≥ 1 Cr(U). (71)

 8log(d)
Proof. The proof is the same as that in Theorem 12, which we omit here. 
6. CONCLUDING REMARKS
In this work, we investigated the connection between circuit complexity and influence, magic, and coherence in quantum circuits. Our main result is a lower bound on the circuit complexity by the circuit sensitivity, magic power, and co- hering power of the circuit.
We provided a characterization of scrambling in quantum circuits by the av- erage sensitivity. We gave a characterization of unitaries with zero circuit sensi- tivity and showed that such unitaries can be efficiently simulated on a classical computer. In other words, circuits consisting of just these unitaries can yield no quantum advantage. In this regard, our result provides a new understanding of matchgates via sensitivity. This raises the following interesting question: does the sensitivity of a quantum circuit determine the classical simulation time of the cir- cuit? This is a question we leave for future work. Moreover, it will be interesting to develop a framework of quantum algorithmic stability based on sensitivity and apply it to quantum differential privacy and generalization capability of quantum machine learning.
Finally, we also defined a quantum version of the Fourier entropy-influence conjecture, and applied it to establishing a connection between circuit complexity and magic. If the quantum Fourier entropy-influence conjecture is true, then we can infer that the classical conjecture also holds.

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 31
ACKNOWLEDGMENTS
K.B. thanks Xun Gao for useful discussions. This work was supported in part by the ARO Grant W911NF-19- 1-0302, the ARO MURI Grant W911NF-20-1- 0082, including Supplemental Support for Research Trainees (SSRT).
APPENDIX A. OTOCS
Lemma 51 (4-point correlator, weight m). If the region D is the last k-th qubit,
then
EAEOA⟨OD(t)OAOD(t)OA⟩ =  n−k ∑ −3 m− j I[n−k][OD(t)],
(72)
where EA denotes the average over all of the size-m subsets A ∈ [n − k] so that OA commutes with OD at the beginning, EOA denotes the average over all local
∑ IS[OD(t)], S⊂[n−k],|S|= j
Pauli operators with support on S is equal to
E P S ⟨ P ⃗a P j P ⃗ b P j ⟩ =  − 1  | s u p p ( ⃗a ) ∩ S | δ ⃗a , ⃗ b .
(73) ( 7 4 )
( 7 5 )
1 m  4jn−k−j(j) m j=0
  PaulioperatorswithweightmonpositionA,andI(j) [OD(t)]isdefinedas [n−k]
I( j) [OD(t)] = [n−k]
POD(t)[⃗a] POD(t)[⃗a]
1 1 m  1  | s u p p ( ⃗a ) ∩ S | =3m n−k∑∑3 −3 POD(t)[⃗a]
∑ P O D ( t ) [ ⃗a ] . S⊂supp(⃗a)
I S [ O D ( t ) ] =
Proof. Let S be a subset of [n − k] with |S| = m. The average of all the weight-m
 Hence,
1  1  | s u p p ( ⃗a ) ∩ S |
EAEOA ⟨OD(t)OAOD(t)OA⟩ =  n−k ∑ ∑ −3
3
  m S⊂[n−k]⃗a
1  1  | s u p p ( ⃗a ) ∩ S |
=  n−k ∑ ∑ −3 m S⊂[n−k]⃗a
     m ⃗a S⊂[n−k]
1 1 ∑ m ∑m  1  j  | s u p p ( ⃗a ) ∩ [ n − k ] | 
= 3m n−k 3 −3 j m ⃗a j=0
   ×n−k−|supp(⃗a)∩[n−k]|PO (t)[⃗a] m−j D

32 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
Let us introduce the Krawtchouk polynomial Km(x;n,q), which is defined as fol- lows:
m ∑m j m−jxn−x
K (x;n,q)= (−1)(q−1) j m−j . (76)
j=0 This can be rewritten as
m ∑m j m−jn−jx
K(x;n,q)= (−q)(q−1) m−j j. (77)
j=0 Then the above equation equals
EAEOA ⟨OD(t)OAOD(t)OA⟩
= 3m  n−k K (|supp(⃗a)∩[n−k]|;n−k,4)P D [⃗a]
1 1 ∑m O(t) m ⃗a
  1 1 ∑∑m j m−jn−k−j|supp(⃗a)∩[n−k]| O (t) =3m n−k (−4)3 m−j j PD [⃗a]
  m ⃗aj=0
1 ∑m  4  j  n − k − j 
=  n−kj=0 −3 m−j m
  |supp(⃗a) ∩ [n − k]|POD(t)[⃗a] j
× ∑ ∑ P O D ( t ) [ ⃗a ] S:S⊂[n−k],|S|= j ⃗a:S⊂supp(⃗a)
× ∑ ⃗a:|supp(⃗a)∩[n−k]|)≥ j
1 ∑m  4  j  n − k − j  =  n−kj=0 −3 m−j
m
1 ∑m  4  j  n − k − j  j D
=  n−k j=0 −3 m− j I[n−k][O (t)].
m
    
= ∥OD(t)∗OD(t)∥2"1−4 1 ∑ Ij[OD(t)∗OD(t)]#, (78) 3n−k j∈[n−k]
where EA denotes the average over all of the positions j ∈ [n − k] so that OA commutes with OD at the beginning, and EOA denotes the average over all local non-identity Pauli operators on position D. The convolution OD(t) ∗ OD(t) is defined in (92).
Lemma 52 (8-point correlator). If the region D is the last k-th qubit, then EAEOA ⟨OD(t)OAOD(t)OAOD(t)OAOD(t)OA⟩
  
COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 33 Proof. Let us express the operator O = ∑⃗a fˆ(⃗a)P⃗a. Then, the correspond general-
ized Wigner function f is defined as follows
f (⃗x) = ∑ fˆ(⃗a)(−1)⟨⃗x,⃗a⟩s , (79) ⃗a
where the inner product ⟨·, ·⟩s denotes the symplectic inner product. (See Appen- dix C for a brief introduction of the generalized Wigner function and symplectic Fourier transformation.)
Let us first consider the average of ⟨P⃗aPjP⃗ PjP⃗cPjP⃗Pj⟩. It is easy to verify that bd
EP ⟨P⃗aPjP⃗ PjP⃗cPjP⃗Pj⟩ jbd
= 1  ⟨ P ⃗a X j P ⃗ X j P ⃗c X j P ⃗ X j ⟩ + ⟨ P ⃗a Y j P ⃗ Y j P ⃗c Y j P ⃗ Y j ⟩ + ⟨ P ⃗a Z j P ⃗ Z j P ⃗c Z j P ⃗ Z j ⟩  3bdbdbd
= 1 ( 4 δ b j + d j , 0 − 1 ) δ ⃗a + ⃗b + ⃗c + d⃗ , ⃗0 3
=  1 − 4 | b j + d j |  δ ⃗a + ⃗b + ⃗c + d⃗ , ⃗0 . 3
Therefore, we have
   EAEOA ⟨OD(t)OAOD(t)OAOD(t)OAOD(t)OA⟩ 14
ˆˆ⃗ˆˆ⃗ = n−k ∑ ∑ 1−3|bj+dj| δ⃗a+⃗b+⃗c+d⃗,⃗0fOD(⃗a)fOD(b)fOD(⃗c)fOD(d)
  j ∈ [ n − k ] ⃗a , ⃗b , ⃗c , d⃗
= n−k ∑ ∑ δ⃗a+⃗b+⃗c+d⃗,⃗0fOD(⃗a)fOD(b)fOD(⃗c)fOD(d)
1
ˆˆ⃗ˆˆ⃗ ˆˆ⃗ˆˆ⃗
 =
ˆˆ⃗ˆˆ⃗ ∑ δ⃗a+⃗b+⃗c+d⃗,⃗0fOD(⃗a)fOD(b)fOD(⃗c)fOD(d)
j ∈ [ n − k ] ⃗a , ⃗b , ⃗c , d⃗
j ∈ [ n − k ] ⃗a , ⃗b , ⃗c , d⃗ 41
−3n−k ∑ ∑ |bj+dj|δ⃗a+⃗b+⃗c+d⃗,⃗0fOD(⃗a)fOD(b)fOD(⃗c)fOD(d) j ∈ [ n − k ] ⃗a , ⃗b , ⃗c , d⃗
  ⃗a,⃗b,⃗c,d⃗ 41
ˆˆ⃗ˆˆ⃗ −3n−k ∑ ∑ |bj+dj|δ⃗a+⃗b+⃗c+d⃗,⃗0fOD(⃗a)fOD(b)fOD(⃗c)fOD(d).
  
34 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE Let us compute the two terms separately. First,
Besides, 1
n−k 1
ˆˆ⃗ˆˆ⃗
h ⃗⃗i ˆˆ⃗ˆˆ⃗
= n−k ∑ ⃗a,⃗b,⃗c,d⃗
[n−k]∩ b+d δ⃗a+⃗b+⃗c+d⃗,⃗0 fOD(⃗a)fOD(b)fOD(⃗c)fOD(d) ˆˆ⃗ˆˆ⃗
ˆˆ⃗ˆˆ⃗ ∑ δ⃗a+⃗b+⃗c+d⃗,⃗0fOD(⃗a)fOD(b)fOD(⃗c)fOD(d)
⃗a,⃗b,⃗c,d⃗ ˆˆ⃗ˆˆ⃗
= ∑ f(⃗a)fOD(⃗a+b)fOD(⃗a+⃗c)fOD(⃗a+b+⃗c) ⃗a,⃗b,⃗c
"ˆ ˆ ⃗#"ˆ ˆ ⃗# = ∑ ∑fOD(⃗a)fOD(⃗a+b) f (⃗c)f (⃗c+b)
⃗b ⃗a
 2
= ∑ E⃗a|fOD(⃗a)| (−1)
⃗b
= E ⃗a | f O D ( ⃗a ) | 4
= ∥OD∗OD∥2, where the convolution O ∗ P satisfies
∑ OD OD ⃗c
⟨⃗a,⃗b⟩ 2 s
fO∗P = fO fP.
∑ ∑ |bj +dj|δ⃗a+⃗b+⃗c+d⃗,⃗0 fOD(⃗a)fOD(b)fOD(⃗c)fOD(d)
(80)
 j ∈ [ n − k ] ⃗a , ⃗b , ⃗c , d⃗
 1
= n−k ∑|[n−k]∩⃗c|fOD(⃗a)fOD(⃗a+b)fOD(⃗a+⃗c)fOD(⃗a+b+⃗c) ⃗a,⃗b,⃗c
 "ˆˆ #"ˆ⃗ˆ⃗# = n−k∑|[n−k]∩⃗c| ∑fOD(⃗a)fOD(⃗a+⃗c) f (b)f (b+⃗c)
1
∑ OD OD ⃗c ⃗a ⃗b
  2 ⟨⃗a,⃗c⟩ 2 n − k ⃗c
1
= ∑|[n−k]∩⃗c| E⃗a|fOD(⃗a)| (−1) s ,
 where it is easy to verify that
Ij[O∗O]= ∑ E⃗a|fO (⃗a)|2(−1)⟨⃗a,⃗c⟩s 2. (81)

Lemma 53 (8-point correlator, weight m). If the region D is the last k-th qubit, then
(82)
D ⃗c:c j ̸=0
EAEOA ⟨OD(t)OAOD(t)OA⟨OD(t)OAOD(t)OA⟩
D D21∑m4jn−k−j(j)D D
= ∥O (t)∗O (t)∥2  n−k j=0 −3 m− j I[n−k][O (t)∗O (t)], m
  
COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 35
where EA denotes the average over all of the size-m subsets A ∈ [n − k] so that OA commutes with OD at the beginning, EOA denotes the average over all local
Pauli operators with weight m on position A, and I(j) [OD(t)∗OD(t)] is defined
bd
[n−k]
Proof. Since O = ∑⃗a fˆ(⃗a)P⃗a, let us first consider the average of ⟨P⃗aPSP⃗ PSP⃗cPSP⃗PS⟩.
as above.
It is easy to verify that
E P S ⟨ P ⃗a P S P ⃗ P S P ⃗c P S P ⃗ P S ⟩ =  − 1  | s u p p ( ⃗b + d⃗ ) ∩ S | δ ⃗a + ⃗b + ⃗c + d⃗ , ⃗0 . ( 8 3 )
Hence, we have
EAEOA ⟨OD(t)OAOD(t)OAOD(t)OAOD(t)OA⟩
 bd3
 1  | s u p p (⃗b + d⃗ ) ∩ S |
− 3 δ ⃗a + ⃗b + ⃗c + d⃗ , 0 f O D ( ⃗a ) f O D ( b ) f O D ( ⃗c ) f O D ( d )
1
=   n − k  m
m  1  | s u p p (⃗b + d⃗ ) ∩ S | =3m n−k∑∑3 −3 δ⃗a+⃗b+⃗c+d⃗,0
m ⃗a , ⃗b , ⃗c , d⃗ S ⊂ [ n − k ]
1 1 = 3m n−k
m
∑ ∑
ˆˆ⃗ˆˆ⃗
  S ⊂ [ n − k ] ⃗a , ⃗b , ⃗c , d⃗ 1 1
   × fˆOD (⃗a) fˆOD (⃗b) fˆOD (⃗c) fˆOD (d⃗)
∑ m∑m  1j|[n−k]∩supp(⃗b+d⃗)|
3 −3 j ⃗a,⃗b,⃗c,d⃗ j=0
   ×n−k−|[n−k]∩supp(⃗b+d⃗)| m−j
×δ fˆ (⃗a) fˆ (⃗b) fˆ (⃗c) fˆ (d⃗) ⃗a+⃗b+⃗c+d⃗,0 OD OD OD OD
11∑m ⃗⃗
= 3m  n−k K (|[n−k]∩supp(b+d)|;n−k,4)
  m ⃗a,⃗b,⃗c,d⃗
×δ fˆ (⃗a) fˆ (⃗b) fˆ (⃗c) fˆ (d⃗)
⃗a+⃗b+⃗c+d⃗,0 OD OD OD OD = 3m  n−k K (|[n−k]∩supp(⃗c)|;n−k,4)
11∑m m ⃗a,⃗b,⃗c
  × fˆOD (⃗a) fˆOD (⃗a +⃗b) fˆOD (⃗a +⃗c) fˆOD (⃗a +⃗b +⃗c) = 3m  n−k K (|[n−k]∩supp(⃗c)|;n−k,4)
ˆˆ⃗ˆˆ⃗
× ∑ f O D ( ⃗a ) f O D ( ⃗a + b ) f O D ( ⃗a + ⃗c ) f O D ( ⃗a + b + ⃗c ) .
⃗a,⃗b
Since
f ˆ ( ⃗a ) f ˆ O D ( ⃗a + ⃗b ) f ˆ O D ( ⃗a + ⃗c ) f ˆ O D ( ⃗a + ⃗b + ⃗c ) =  E ⃗a | f ( ⃗a ) | 2 ( − 1 ) ⟨ ⃗a , ⃗c ⟩ s  2 ,
11∑m m ⃗c
  ( 8 4 )

36 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE we have
EAEOA ⟨OD(t)OAOD(t)OAOD(t)OAOD(t)OA⟩ = 3m  n−k K (|[n−k]∩supp(⃗c)|;n−k,4)
ˆˆ⃗ˆˆ⃗
× ∑ f O D ( ⃗a ) f O D ( ⃗a + b ) f O D ( ⃗a + ⃗c ) f O D ( ⃗a + b + ⃗c )
⃗a,⃗b
   K (|[n−k]∩supp(⃗c)|;n−k,4) E |f(⃗a)| (−1) s 1 1 ∑m ⃗a 2

In this section, we give a brief introduction to the Fourier entropy-influence conjecture for Boolean functions. Boolean functions, defined as functions f : { −1, 1 }n → { −1, 1 } (or R), are a basic object in theoretical computer science. The inner product between Boolean functions is defined as
⟨f,g⟩ := Ex f(x)g(x),
where E := 1 ∑ n . Each Boolean function f has the following Fourier
f(x)= ∑ fˆ(S)xS, S⊂[n]
where the parity functions xS := ∏i∈S xi, and the Fourier coefficients fˆ(S) = ⟨ f , xS ⟩ = Ex∈{ −1,1 }n f (x)xS . Parseval’s identity tells us that
Ex∈{±}n f(x)2 =∑fˆ(S)2. S
Let us define the discrete derivative Dj[f] as Dj[f](x) = (f(x)− f(x⊕ej))/2, where x⊕ej denotes the flip from xj to −xj. Then the j-th local influence Ij is defined as the l2 norm of the discrete derivative Di[ f ]:
Ij[f] = Ex∈{±1}n|Dj[f](x)|2, which can also be written as
Ij[f]= ∑ fˆ(S)2|S|, S: j∈S
11∑m m ⃗c
  =
= 3m n−k∑∑(−4)3 m−j j |fO∗O(⃗c)|
⟨⃗a,⃗c⟩2
  3m n−k m
⃗c
1 1 m j m−jn−k−j|[n−k]∩supp(⃗c)| ˆ
2
  m ⃗cj=0
D D21∑m4jn−k−jj D D
= ∥O (t)∗O (t)∥2  n−k j=0 −3 m− j I[n−k][O (t)∗O (t)]. m
  APPENDIX B. BOOLEAN FOURIER ENTROPY-INFLUENCE CONJECTURE
x 2n x∈{±} expansion
 
COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 37 where |S| denotes the size of the subset S. The total influence of the Boolean
function is defined as I[f] = ∑j∈[n] Ij[f], which can also be written as I[f]= ∑ fˆ(S)2|S|.
S⊂[n]
Assume that ∥ f ∥2 = 1. Then, ∑S fˆ(S)2 = 1 and the Fourier entropy of the
Boolean function f is defined as H[f]=∑|fˆ(S)|2log 1 ,
 S⊂[n] | fˆ(S)|2 and the min Fourier entropy H∞ is defined as
H∞[f]=minlog 1 . S⊂[n] | fˆ(S)|2
One of most important open problems in the analysis of Boolean functions is proving the Fourier entropy-influence (FEI) conjecture that was proposed by Friedgut and Kalai [52].
Conjecture 54 (FEI conjecture). There exists a universal constant c such that, for all f : {−1,1}n → {−1,1},
H[f]≤cI[f]. (85) A natural extension of the FEI conjecture is the following Fourier min-entropy-
influence conjecture, which follows from the fact that Hmin[ f ] ≤ H[ f ]. Conjecture 55 (FMEI conjecture). There exists a universal constant c such that,
for all f : {−1,1}n → {−1,1},
Hmin[f]≤cI[f]. (86)
Although both the FEI and FMEI conjectures remain open, several significant steps have been made to prove these conjectures; see [81–88].
APPENDIX C. DISCRETE WIGNER FUNCTION AND SYMPLECTIC FOURIER TRANSFORMATION
We introduce some basics on the Fourier analysis of the discrete Wigner func- tion. The discrete Wigner function was proposed for the odd-dimensional case, and one well-known result for odd-dimensional discrete Wigner functions is the discrete Hudson theorem, which states that any given pure state is a stabilizer state if and only if its Wigner function is nonnegative [89]. Here, we generalize the definition of the discrete Wigner function to the qubit case, where the discrete Hudson theorem may not hold.
Let us define the generalized phase point operator as follows
A⃗a =∑P⃗(−1)⟨⃗a,⃗b⟩s, (87)
b ⃗b
 
38 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
where P⃗b is an n-qubit Pauli operator and ⟨⃗a,⃗b⟩s denotes the symplectic inner product. Hence, given an observable O (or a quantum state), the (generalized) discrete Wigner function f is defined as follows
f O ( ⃗a ) = ⟨ O , A ⃗a ⟩ , which can also be written as follows
fO(⃗a) = ∑⟨P⃗ ,O⟩(−1)⟨⃗a,⃗b⟩s = ∑O⃗ (−1)⟨⃗a,⃗b⟩s. bb
( 8 8 ) (89)
⃗b ⃗b
Hence, the Pauli coefficient O⃗b is the symplectic Fourier transform of the discrete
Wigner function, i.e.,
O = fˆ (⃗b)=E f (⃗a)(−1)⟨⃗a,⃗b⟩s. (90) ⃗bO ⃗aO
To consider the higher-order OTOC, we need to use the convolution of two observables. We define the convolution of two observables O1 and O2 as follows
Parseval’s identity tells us that 2ˆ⃗22
Hence
fO1∗O2 = fO1 fO2. (92) f ˆ O 1 ∗ O 2 ( ⃗b ) = E ⃗a f O 1 ( ⃗a ) f O 2 ( ⃗a ) ( − 1 ) ⟨ ⃗a , ⃗b ⟩ s . ( 9 3 )
REFERENCES
E⃗af(⃗a) =∑fO(b) =∑|O⃗b|. (91) ⃗b ⃗b
[1] M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information. Cam- bridge University Press, 2010.
[2] A. Kitaev, A. Shen, and M. Vyalyi, Classical and quantum computation. American Mathe- matical Society, 2002.
[3] S.Aaronson,“Thecomplexityofquantumstatesandtransformations:fromquantummoney to black holes,” arXiv preprint arXiv:1607.05256, 2016.
[4] M. A. Nielsen, “A geometric approach to quantum circuit lower bounds,” Quantum Infor- mation & Computation, vol. 6, no. 3, pp. 213–262, 2006.
[5] M. A. Nielsen, M. R. Dowling, M. Gu, and A. C. Doherty, “Optimal control, geometry, and quantum computing,” Phys. Rev. A, vol. 73, p. 062323, Jun 2006.
[6] M. A. Nielsen, M. R. Dowling, M. Gu, and A. C. Doherty, “Quantum computation as ge- ometry,” Science, vol. 311, no. 5764, pp. 1133–1135, 2006.
[7] M. R. Dowling and M. A. Nielsen, “The geometry of quantum computation,” Quantum Information & Computation, vol. 8, no. 10, pp. 861–899, 2008.
[8] A. R. Brown, L. Susskind, and Y. Zhao, “Quantum complexity and negative curvature,” Phys. Rev. D, vol. 95, p. 045010, Feb 2017.
[9] L.Susskind,“Thetypical-stateparadox:diagnosinghorizonswithcomplexity,”Fortschritte der Physik, vol. 64, no. 1, pp. 84–91, 2016.
[10] A. R. Brown, D. A. Roberts, L. Susskind, B. Swingle, and Y. Zhao, “Holographic complex- ity equals bulk action?,” Phys. Rev. Lett., vol. 116, p. 191301, May 2016.
[11] S. Chapman, M. P. Heller, H. Marrochio, and F. Pastawski, “Toward a definition of com- plexity for quantum field theory states,” Phys. Rev. Lett., vol. 120, p. 121602, Mar 2018.
[12] F. G. Brandão, W. Chemissany, N. Hunter-Jones, R. Kueng, and J. Preskill, “Models of quantum complexity growth,” PRX Quantum, vol. 2, p. 030316, Jul 2021.

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 39
[13] R. A. Jefferson and R. C. Myers, “Circuit complexity in quantum field theory,” Journal of High Energy Physics, vol. 2017, no. 10, pp. 1–81, 2017.
[14] T. Takayanagi, “Holographic spacetimes as quantum circuits of path-integrations,” Journal of High Energy Physics, vol. 2018, no. 12, pp. 1–37, 2018.
[15] A. Bhattacharyya, A. Shekar, and A. Sinha, “Circuit complexity in interacting qfts and rg flows,” Journal of High Energy Physic, vol. 2018, p. 140, Oct 2018.
[16] N. Chagnet, S. Chapman, J. de Boer, and C. Zukowski, “Complexity for conformal field theories in general dimensions,” Phys. Rev. Lett., vol. 128, p. 051601, Jan 2022.
[17] A. Bhattacharyya, G. Katoch, and S. R. Roy, “Complexity of warped conformal field the- ory,” arXiv preprint arXiv:2202.09350, 2022.
[18] J. Couch, Y. Fan, and S. Shashi, “Circuit complexity in topological quantum field theory,” arXiv preprint arXiv:2108.13427, 2021.
[19] N. Y. Halpern, N. B. Kothakonda, J. Haferkamp, A. Munson, J. Eisert, and P. Faist, “Re- source theory of quantum uncomplexity,” arXiv preprint arXiv:2110.11371, 2021.
[20] J. Eisert, “Entangling power and quantum circuit complexity,” Phys. Rev. Lett., vol. 127, p. 020501, Jul 2021.
[21] R. O’Donnell, Analysis of Boolean functions. Cambridge University Press, 2014.
[22] J. Kahn, G. Kalai, and N. Linial, “The influence of variables on Boolean functions,” in [Proceedings 1988] 29th Annual Symposium on Foundations of Computer Science, pp. 68–
80, 1988.
[23] N. Linial, Y. Mansour, and N. Nisan, “Constant depth circuits, Fourier transform, and learn-
ability,” in 30th Annual Symposium on Foundations of Computer Science, pp. 574–579,
1989.
[24] R.B.Boppana,“Theaveragesensitivityofbounded-depthcircuits,”InformationProcessing
Letters, vol. 63, no. 5, pp. 257–261, 1997.
[25] S. Jukna, Boolean Function Complexity: Advances and Frontiers. Berlin, Germany:
Springer, 2012.
[26] S. Lovett and E. Viola, “Bounded-depth circuits cannot sample good codes,” in 2011 IEEE
26th Annual Conference on Computational Complexity, pp. 243–251, 2011.
[27] Y.Shi,“Lowerboundsofquantumblack-boxcomplexityanddegreeofapproximatingpoly- nomials by influence of Boolean variables,” Information Processing Letters, vol. 75, no. 1,
pp. 79–83, 2000.
[28] A. Montanaro and T. J. Osborne, “Quantum boolean functions,” Chicago Journal of Theo-
retical Computer Science, vol. 2010, January 2010.
[29] E. A. Carlen and E. H. Lieb, “Optimal hypercontractivity for Fermi fields and related non-
commutative integration inequalities,” Communications in Mathematical Physics, vol. 155,
no. 1, pp. 27 – 46, 1993.
[30] L.G.Valiant,“Quantumcircuitsthatcanbesimulatedclassicallyinpolynomialtime,”SIAM
Journal on Computing, vol. 31, no. 4, pp. 1229–1254, 2002.
[31] S. Bravyi, “Lagrangian representation for fermionic linear optics,” Quantum Information &
Computation, vol. 5, no. 3, pp. 216–238, 2005.
[32] D. P. DiVincenzo and B. M. Terhal, “Fermionic linear optics revisited,” Foundations of
Physics, vol. 35, pp. 1967–1984, 2004.
[33] B. M. Terhal and D. P. DiVincenzo, “Classical simulation of noninteracting-fermion quan-
tum circuits,” Phys. Rev. A, vol. 65, p. 032325, Mar 2002.
[34] R. Jozsa and A. Miyake, “Matchgates and classical simulation of quantum circuits,” Proc.
R. Soc. Lond. A, vol. 464, p. 3089–3106, Jul 2008.
[35] D. J. Brod, “Efficient classical simulation of matchgate circuits with generalized inputs and
measurements,” Phys. Rev. A, vol. 93, p. 062332, Jun 2016.
[36] M. Hebenstreit, R. Jozsa, B. Kraus, S. Strelchuk, and M. Yoganathan, “All pure fermionic
non-Gaussian states are magic states for matchgate computations,” Phys. Rev. Lett., vol. 123, p. 080503, Aug 2019.

40 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
[37] D.Gottesman,“TheHeisenbergrepresentationofquantumcomputers,”inProc.XXIIInter- national Colloquium on Group Theoretical Methods in Physics, 1998, pp. 32–43, 1998.
[38] M. V. den Nest, “Classical simulation of quantum computation, the Gottesman-Knill theo- rem, and slightly beyond,” Quantum Information & Computation, vol. 10, no. 3-4, pp. 0258– 0271, 2010.
[39] R. Jozsa and M. Van den Nest, “Classical simulation complexity of extended Clifford cir- cuits,” Quantum Information & Computation, vol. 14, no. 7&8, pp. 633–648, 2014.
[40] D. E. Koh, “Further extensions of Clifford circuits and their classical simulation complexi- ties,” Quantum Information & Computation, vol. 17, no. 3&4, pp. 262–282, 2017.
[41] A.Bouland,J.F.Fitzsimons,andD.E.Koh,“ComplexityClassificationofConjugatedClif- ford Circuits,” in 33rd Computational Complexity Conference (CCC 2018) (R. A. Servedio, ed.), vol. 102 of Leibniz International Proceedings in Informatics (LIPIcs), (Dagstuhl, Ger- many), pp. 21:1–21:25, Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2018.
[42] M. Yoganathan, R. Jozsa, and S. Strelchuk, “Quantum advantage of unitary Clifford cir- cuits with magic state inputs,” Proceedings of the Royal Society A, vol. 475, no. 2225, p. 20180427, 2019.
[43] S. Bravyi, G. Smith, and J. A. Smolin, “Trading classical and quantum computational re- sources,” Phys. Rev. X, vol. 6, p. 021043, Jun 2016.
[44] S. Bravyi, D. Browne, P. Calpin, E. Campbell, D. Gosset, and M. Howard, “Simulation of quantum circuits by low-rank stabilizer decompositions,” Quantum, vol. 3, p. 181, Sept. 2019.
[45] M. Howard and E. Campbell, “Application of a resource theory for magic states to fault- tolerant quantum computing,” Phys. Rev. Lett., vol. 118, p. 090501, Mar 2017.
[46] J. R. Seddon, B. Regula, H. Pashayan, Y. Ouyang, and E. T. Campbell, “Quantifying quan- tum speedups: Improved classical simulation from tighter magic monotones,” PRX Quan- tum, vol. 2, p. 010345, Mar 2021.
[47] J. R. Seddon and E. T. Campbell, “Quantifying magic for multi-qubit operations,” Proc. R. Soc. A., vol. 475, 2019.
[48] X. Wang, M. M. Wilde, and Y. Su, “Quantifying the magic of quantum channels,” New Journal of Physics, vol. 21, p. 103002, Oct 2019.
[49] K. Bu and D. E. Koh, “Efficient classical simulation of Clifford circuits with nonstabilizer input states,” Phys. Rev. Lett., vol. 123, p. 170502, Oct 2019.
[50] K. Bu and D. E. Koh, “Classical simulation of quantum circuits by half Gauss sums,” Com- mun. Math. Phys., vol. 390, pp. 471–500, Mar 2022.
[51] Z.-W. Liu and A. Winter, “Many-body quantum magic,” arXiv preprint arXiv:2010.13817, 2020.
[52] E.FriedgutandG.Kalai.,“Everymonotonegraphpropertyhasasharpthreshold,”Proceed- ings of the American mathematical Society, vol. 124, pp. 2993–3002, Aug 1996.
[53] Y. Mansour, “Learning Boolean functions via the Fourier transform,” in Theoretical ad- vances in neural computation and learning, pp. 391–424, Springer, 1994.
[54] M. Lostaglio, K. Korzekwa, D. Jennings, and T. Rudolph, “Quantum coherence, time- translation symmetry, and thermodynamics,” Phys. Rev. X, vol. 5, p. 021001, Apr 2015.
[55] M. Lostaglio, D. Jennings, and T. Rudolph, “Description of quantum coherence in thermo- dynamic processes requires constraints beyond free energy,” Nature communications, vol. 6, no. 1, pp. 1–9, 2015.
[56] M. B. Plenio and S. F. Huelga, “Dephasing-assisted transport: quantum networks and biomolecules,” New Journal of Physics, vol. 10, p. 113019, Nov 2008.
[57] S. Lloyd, “Quantum coherence in biological systems,” Journal of Physics: Conference Se- ries, vol. 302, p. 012037, Jul 2011.
[58] F. Levi and F. Mintert, “A quantitative theory of coherent delocalization,” New Journal of Physics, vol. 16, p. 033007, Mar 2014.
[59] J. Aberg, “Quantifying superposition,” arXiv preprint quant-ph/0612146, 2006.

COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE 41
[60] T. Baumgratz, M. Cramer, and M. B. Plenio, “Quantifying coherence,” Phys. Rev. Lett., vol. 113, p. 140401, Sep 2014.
[61] A. Winter and D. Yang, “Operational resource theory of coherence,” Phys. Rev. Lett., vol. 116, p. 120404, Mar 2016.
[62] K.Bu,U.Singh,S.-M.Fei,A.K.Pati,andJ.Wu,“Maximumrelativeentropyofcoherence: An operational coherence measure,” Phys. Rev. Lett., vol. 119, p. 150405, Oct 2017.
[63] A.Streltsov,G.Adesso,andM.B.Plenio,“Colloquium:Quantumcoherenceasaresource,” Rev. Mod. Phys., vol. 89, p. 041003, Oct 2017.
[64] F.Bischof,H.Kampermann,andD.Bruß,“Resourcetheoryofcoherencebasedonpositive- operator-valued measures,” Phys. Rev. Lett., vol. 123, p. 110402, Sep 2019.
[65] M. Mariën, K. M. Audenaert, K. Van Acoleyen, and F. Verstraete, “Entanglement rates and the stability of the area law for the entanglement entropy,” Communications in Mathematical Physics, vol. 346, no. 1, pp. 35–73, 2016.
[66] C. Dwork and A. Roth, “The Algorithmic Foundations of Differential Privacy,” Found. Trends. Theor. Comput. Sci., vol. 9, pp. 211–407, Aug. 2014.
[67] C.Dwork,F.McSherry,K.Nissim,andA.Smith,“Calibratingnoisetosensitivityinprivate data analysis,” Journal of Privacy and Confidentiality, vol. 7, no. 3, pp. 17–51, 2016.
[68] O.BousquetandA.Elisseeff,“Stabilityandgeneralization,”TheJournalofMachineLearn-
ing Research, vol. 2, pp. 499–526, 2002.
[69] O.Bousquet,Y.Klochkov,andN.Zhivotovskiy,“Sharperboundsforuniformlystablealgo-
rithms,” in Proceedings of Thirty Third Conference on Learning Theory (J. Abernethy and S. Agarwal, eds.), vol. 125 of Proceedings of Machine Learning Research, pp. 610–626, PMLR, 09–12 Jul 2020.
[70] L. Zhou and M. Ying, “Differential privacy in quantum computation,” in 2017 IEEE 30th Computer Security Foundations Symposium (CSF), pp. 249–262, 2017.
[71] S. Aaronson and G. N. Rothblum, “Gentle measurement of quantum states and differen- tial privacy,” in Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, STOC 2019, (New York, NY, USA), pp. 322–333, Association for Computing Machinery, 2019.
[72] L. Banchi, J. Pereira, and S. Pirandola, “Generalization in quantum machine learning: A quantum information standpoint,” PRX Quantum, vol. 2, p. 040321, Nov 2021.
[73] M. C. Caro, H.-Y. Huang, M. Cerezo, K. Sharma, A. Sornborger, L. Cincio, and P. J. Coles, “Generalization in quantum machine learning from few training data,” arXiv preprint arXiv:2111.05292, 2021.
[74] K. Bu, D. E. Koh, L. Li, Q. Luo, and Y. Zhang, “On the statistical complexity of quantum circuits,” arXiv preprint arXiv:2101.06154, 2021.
[75] K. Bu, D. E. Koh, L. Li, Q. Luo, and Y. Zhang, “Effects of quantum resources on the statistical complexity of quantum circuits,” arXiv preprint arXiv:2102.03282, 2021.
[76] K. Bu, D. E. Koh, L. Li, Q. Luo, and Y. Zhang, “Rademacher complexity of noisy quantum circuits,” arXiv preprint arXiv:2103.03139, 2021.
[77] M. C. Caro, H.-Y. Huang, N. Ezzell, J. Gibbs, A. T. Sornborger, L. Cincio, P. J. Coles, and Z. Holmes, “Out-of-distribution generalization for learning quantum dynamics,” arXiv preprint arXiv:2204.10268, 2022.
[78] J. Gibbs, Z. Holmes, M. C. Caro, N. Ezzell, H.-Y. Huang, L. Cincio, A. T. Sornborger, and P. J. Coles, “Dynamical simulation via quantum machine learning with provable generaliza- tion,” arXiv preprint arXiv:2204.10269, 2022.
[79] S. Aaronson and D. Gottesman, “Improved simulation of stabilizer circuits,” Phys. Rev. A, vol. 70, p. 052328, Nov 2004.
[80] K. M. Audenaert, “Quantum skew divergence,” Journal of Mathematical Physics, vol. 55, no. 11, p. 112202, 2014.
[81] E. Kelman, G. Kindler, N. Lifshitz, D. Minzer, and M. Safra, “Towards a proof of the Fourier-entropy conjecture?,” Geometric and Functional Analysis, vol. 30, pp. 1097–1138, Aug 2020.

42 COMPLEXITY OF QUANTUM CIRCUITS VIA SENSITIVITY, MAGIC, AND COHERENCE
[82] S. Chakraborty, R. Kulkarni, S. V. Lokam, and N. Saurabh, “Upper bounds on Fourier en- tropy,” Theoretical Computer Science, vol. 654, pp. 92–112, 2016. Computing and Combi- natorics.
[83] A. R. Klivans, H. K. Lee, and A. Wan, “Mansour’s conjecture is true for random DNF formulas,” in COLT, pp. 368–380, Citeseer, 2010.
[84] R. O’Donnell and L.-Y. Tan, “A composition theorem for the Fourier entropy-influence conjecture,” in Automata, Languages, and Programming (F. V. Fomin, R. Freivalds, M. Kwiatkowska, and D. Peleg, eds.), (Berlin, Heidelberg), pp. 780–791, Springer Berlin Heidelberg, 2013.
[85] R. O’Donnell, J. Wright, and Y. Zhou, “The Fourier entropy–influence conjecture for cer- tain classes of Boolean functions,” in Automata, Languages and Programming (L. Aceto, M. Henzinger, and J. Sgall, eds.), (Berlin, Heidelberg), pp. 330–341, Springer Berlin Hei- delberg, 2011.
[86] G.Shalev,“OntheFourierentropyinfluenceconjectureforextremalclasses,”arXivpreprint arXiv:1806.03646, 2018.
[87] A. Wan, J. Wright, and C. Wu, “Decision trees, protocols and the entropy-influence conjec- ture,” in Proceedings of the 5th Conference on Innovations in Theoretical Computer Science, ITCS ’14, (New York, NY, USA), p. 67–80, Association for Computing Machinery, 2014.
[88] P. Gopalan, R. A. Servedio, and A. Wigderson, “Degree and Sensitivity: Tails of Two Dis- tributions,” in 31st Conference on Computational Complexity (CCC 2016) (R. Raz, ed.), vol. 50 of Leibniz International Proceedings in Informatics (LIPIcs), (Dagstuhl, Germany), pp. 13:1–13:23, Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2016.
[89] D. Gross, “Hudson’s theorem for finite-dimensional quantum systems,” Journal of Mathe- matical Physics, vol. 47, no. 12, p. 122107, 2006.
