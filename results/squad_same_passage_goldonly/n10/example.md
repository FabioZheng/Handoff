# Example: question-only conditioning difference

**Question A passed only in the conditioned arm:** What is an example of a measurement within a complexity class that would create a bigger set of problems if the bounds were relaxed?

**Unrelated held-out Question B:** In what expression can one expect to find DTIME(n)

One shared SQuAD passage (Computational_complexity_theory) answers both questions, placed at P1 among 9 length-matched SQuAD passages LLM-screened irrelevant to both A and B.

## conditioned, stage 1

**Research Notes**

**Context:** Complexity classes defined by constraining resources (e.g., time, space)

**Key Concept:** Hierarchy theorems induce a proper hierarchy on the classes defined by constraining resources

**Observation:** Relaxing requirements on resources (e.g., computation time) defines a bigger set of problems

**Specific Example:** DTIME(n) is contained in DTIME(n^2)

**Theorem:** Time hierarchy theorem states that for any time complexity class C, there exists a polynomial p such that C is properly contained in DTIME(p(n))

**Implication:** Relaxing time requirements from DTIME(n) to DTIME(n^2) indeed defines a bigger set of problems

**Note:** The time hierarchy theorem provides a general framework for understanding the relationship between time complexity classes, but the specific example of DTIME(n) and DTIME(n^2) is a concrete illustration of the concept.

## generic, stage 1

Research Notes: Complexity Class Hierarchy Theorems

Objective: Investigate the relationship between complexity classes defined by time and space requirements.

Key Findings:

1. The time and space hierarchy theorems provide a framework for understanding the relationship between complexity classes.
2. These theorems induce a proper hierarchy on the classes defined by constraining time and space resources.
3. The hierarchy theorems imply that there are pairs of complexity classes where one is properly included in the other.

Specifically:

- DTIME(n) is contained in DTIME(n^2)
- The relationship between these classes is to be determined.

Next Steps:

1. Investigate the time hierarchy theorem to understand its implications on the relationship between DTIME(n) and DTIME(n^2).
2. Explore the space hierarchy theorem to determine its relevance to the relationship between complexity classes defined by space requirements.

## conditioned, stage 5

**Research Notes**

The hierarchy theorems establish a proper hierarchy among classes defined by resource constraints, such as time and space. This hierarchy indicates that relaxing requirements on resources, e.g., computation time, results in a larger set of problems. A specific example is the containment of DTIME(n) in DTIME(n^2), where the latter class includes problems solvable in time O(n^2) and the former in time O(n).

## generic, stage 5

Research Notes: Complexity Class Hierarchy Theorems

The time and space hierarchy theorems establish a framework for understanding the relationship between complexity classes defined by time and space requirements. These theorems induce a proper hierarchy on the classes defined by constraining time and space resources, implying that there are pairs of complexity classes where one is properly included in the other.

The time hierarchy theorem has significant implications for understanding the relationship between DTIME(n) and DTIME(n^2), where DTIME(n) is contained in DTIME(n^2). This relationship highlights the importance of exploring the space hierarchy theorem to gain insight into the relevance of complexity classes defined by space requirements.

The hierarchy theorems provide a foundation for understanding the complexity class hierarchy, and further research into these theorems will continue to reveal the intricacies of the relationships between complexity classes.
