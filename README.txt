Supplementary Materials
=======================
"The deeper the word, the further it falls"
Matthias Christian Wurm, March 2026

Scripts require: numpy, scipy, wordfreq, pickle
Embeddings: Hamilton et al. 2016 (https://nlp.stanford.edu/projects/histwords/)

Files
-----
S1_main_analysis.py         Main English analysis (225 words, COHA 1850-2000)
                            r=0.528, mediation, fixpoint test

S2_robustness_analyses.py   10-analysis robustness suite:
                            temporal stability, word classes, frequency strata,
                            pairwise contrasts, Procrustes sensitivity,
                            non-linearity, multiple regression,
                            German & French replications

S3_transparency_coding.py   Transparency ratings for 112 derived words
                            with rule: "modern speaker recognizes parts?"
                            Result: transparent words change MORE (d=0.39)

S4_two_system_model.py      Three-regime model (primes/opaque/transparent)
                            Category adds ΔR²=0.01, p=0.075 (marginal)

S5_german_replication.py    German borrowing-depth replication
                            r_partial = -0.17 (borrowing depth fails)

S6_hand_coded_pilot.py      274-word hand-coded analysis (pre-embedding)
                            r=0.646

S7_initial_pilot.py         Initial 131-word pilot study
                            r=0.668 (19 Primwörter)

S8_morpheme_calculator.py   Automated morpheme-based ed estimator
                            (abandoned: too aggressive stripping)

S9_wiktionary_attempt_failed.py  Automated Wiktionary ed estimation
                            (FAILED: r dropped to 0.162, kept hand-coded)

S10_referent_type_test.py   Referent type analysis (225 words: S/E/N)
                            Combined with transparency: ΔR²=0.013, p=0.026

S11_german_anchor_test.py   German transparency anchor test (195 words)
                            KEY FINDING: transparent compounds change LESS
                            in German (d=-0.42, p=0.034), opposite of English
                            Three-group ANOVA: F=14.26, p<10^-5
                            After freq control: ΔR²=0.036, p=0.029
                            → Transparency anchors meaning in German

S12_german_composition_depth.py  Composition-depth vs borrowing-depth
                            comparison for German (120 words)
                            cd does not beat ed; both go negative after
                            frequency control → led to anchor hypothesis

Word list
---------
The complete 272-word list with hand-coded ed values is embedded
in S1_main_analysis.py (lines 71-162).

Transparency ratings for all 112 derived words (ed ≥ 2) are
in S3_transparency_coding.py with justifications for each word.

German transparency coding for 195 words (P/T/O categories)
is in S11_german_anchor_test.py.
