---
aliases:
  - Organic Reaction Mechanisms
  - Mechanisms in Organic Chemistry
---

# Reaction Mechanisms in Organic Synthesis

Reaction mechanisms describe the step-by-step process by which reactants are transformed into products, including the movement of electrons, the breaking and forming of bonds, and the formation of intermediates. Understanding mechanisms is crucial for predicting reaction outcomes, controlling selectivity, and designing new synthetic strategies.

## Fundamental Concepts

### Electron Movement Representation

- **Curved arrows**: Show the movement of electron pairs
- **Fishhook arrows**: Represent single electron movement in radical reactions
- **Electron-pushing formalism**: Electrons move from electron-rich sites to electron-poor sites

### Key Mechanistic Elements

1. **Nucleophiles**: Electron-rich species seeking positive centers
2. **Electrophiles**: Electron-poor species seeking negative centers
3. **Leaving groups**: Groups that depart with the electron pair from the bond
4. **Intermediates**: Transient species formed during reactions

## Common Reaction Mechanisms

### 1. Nucleophilic Substitution

#### SN2 (Substitution Nucleophilic Bimolecular)

```
      Nu⁻
       |
       v
Nu⁻ + C-LG → [Nu---C---LG]‡ → Nu-C + LG⁻
```

- **Characteristics**: Concerted, single-step process
- **Stereochemistry**: Inversion of configuration at stereocenter
- **Rate law**: Rate = k[Substrate][Nucleophile]
- **Favored by**: Primary substrates, strong nucleophiles, aprotic polar solvents
- **Examples**: Alkyl halide substitution with hydroxide

#### SN1 (Substitution Nucleophilic Unimolecular)

```
       Step 1           Step 2
R-LG → R⁺ + LG⁻       R⁺ + Nu⁻ → R-Nu
```

- **Characteristics**: Two-step process with carbocation intermediate
- **Stereochemistry**: Racemization (or partial retention/inversion)
- **Rate law**: Rate = k[Substrate]
- **Favored by**: Tertiary/secondary substrates, weak nucleophiles, protic polar solvents
- **Examples**: Solvolysis of tert-butyl bromide in aqueous ethanol

### 2. Elimination Reactions

#### E2 (Elimination Bimolecular)

```
     H
     |
B: + C-C-LG → B-H + C=C + LG⁻
     |
     H
```

- **Characteristics**: Concerted, single-step process
- **Stereochemistry**: Anti-periplanar arrangement preferred
- **Rate law**: Rate = k[Substrate][Base]
- **Favored by**: Strong bases, higher temperatures
- **Examples**: Dehydrohalogenation of alkyl halides with KOH

#### E1 (Elimination Unimolecular)

```
       Step 1           Step 2
R-LG → R⁺ + LG⁻       R⁺ → Alkene + H⁺
```

- **Characteristics**: Two-step process with carbocation intermediate
- **Stereochemistry**: More substituted alkene favored (Zaitsev's rule)
- **Rate law**: Rate = k[Substrate]
- **Favored by**: Tertiary substrates, weak bases, protic solvents
- **Examples**: Dehydration of alcohols with H₂SO₄

### 3. Addition Reactions

#### Electrophilic Addition to Alkenes

```
       Step 1              Step 2
C=C + E⁺ → E-C-C⁺       E-C-C⁺ + Nu⁻ → E-C-C-Nu
```

- **Characteristics**: Two-step process with carbocation intermediate
- **Regioselectivity**: Markovnikov's rule (electrophile adds to less substituted carbon)
- **Examples**: Addition of HBr to alkenes, hydration of alkenes

#### Nucleophilic Addition to Carbonyls

```
       O                  O⁻
       ‖                  |
R-C-R + Nu⁻ → R-C-R → R-C-R
                |         |
               Nu        Nu
                      (protonation)
```

- **Characteristics**: Addition to the π bond of C=O
- **Stereochemistry**: Formation of tetrahedral intermediate
- **Examples**: Grignard addition, hydride reduction, cyanohydrin formation

### 4. Radical Reactions

#### Radical Chain Mechanism

```
Initiation:     In-In → 2 In·
Propagation:    In· + R-X → In-X + R·
                R· + Y-Z → R-Y + Z·
Termination:    R· + R· → R-R
```

- **Characteristics**: Involves unpaired electrons, three-stage process
- **Regioselectivity**: Governed by radical stability
- **Examples**: Halogenation of alkanes, radical polymerization

### 5. Pericyclic Reactions

#### Cycloaddition (Diels-Alder)

```
     \      /
      \    /
       \  /
        \/
        /\
       /  \
      /    \
     /      \
```

- **Characteristics**: Concerted, single-step process
- **Stereochemistry**: Endo preference in Diels-Alder
- **Orbital considerations**: HOMO-LUMO interactions
- **Examples**: Diels-Alder reaction, 1,3-dipolar cycloadditions

#### Electrocyclic Reactions

```
        conrotatory        disrotatory
          →  ←               → →
     \      /            \      /
      \    /              \    /
       \  /                \  /
```

- **Characteristics**: Concerted electronic reorganization
- **Stereochemistry**: Governed by Woodward-Hoffmann rules
- **Examples**: Ring opening/closing of cyclobutene, cyclohexadiene

#### Sigmatropic Rearrangements

```
[1,5]-H shift:

H                         
|                        H
C==C--C==C--C  →  C==C--C==C--C
```

- **Characteristics**: Migration of σ bond across π system
- **Stereochemistry**: Governed by orbital symmetry
- **Examples**: Cope rearrangement, Claisen rearrangement

### 6. Carbonyl Chemistry

#### Aldol Reaction

```
Base:     R-C-CH₃  →  R-C-CH₂⁻
          ‖            ‖ 
          O            O

          O            OH O
          ‖            |  ‖
R-C-CH₂⁻ + R'-C-H → R-C-CH-C-H
                          |
                          R'
```

- **Characteristics**: Nucleophilic addition followed by dehydration
- **Stereochemistry**: Can be controlled with chiral auxiliaries
- **Examples**: Aldol condensation, Robinson annulation

#### Claisen Condensation

```
          O                   O  O
          ‖                   ‖  ‖
R-C-OCH₃ + CH₃-C-OCH₃ → R-C-CH₂-C-OCH₃
```

- **Characteristics**: Formation of β-keto esters or β-diketones
- **Examples**: Dieckmann condensation, acetoacetic ester synthesis

### 7. Organometallic Reactions

#### Grignard Reaction

```
       MgX              OMgX
       |                |
R-MgX + C=O → R-C-OMgX → R-C-OH
                |         |
                R'        R'
```

- **Characteristics**: Nucleophilic addition to carbonyl
- **Limitations**: Incompatible with acidic protons
- **Examples**: Addition to aldehydes/ketones, formation of alcohols

#### Cross-Coupling Reactions

```
R-X + R'-M → [Catalyst] → R-R' + M-X
```

- **Characteristics**: Transition metal-catalyzed C-C bond formation
- **Catalyst cycle**: Oxidative addition, transmetallation, reductive elimination
- **Examples**: Suzuki, Heck, Sonogashira reactions

## Reaction Energy Profiles

```
       ‡
Energy |     Transition State
       |        /\
       |       /  \
       |      /    \
       |     /      \
       |    /        \
       |   /          \
       |  /            \
       | /              \
       |/                \
       Reactants       Products
       ----------------------
        Reaction Coordinate
```

Key concepts:
- **Transition state**: Highest energy point on reaction pathway
- **Intermediates**: Local energy minima between reactants and products
- **Activation energy**: Energy difference between reactants and transition state
- **Reaction energy**: Energy difference between reactants and products

## Analyzing and Predicting Mechanisms

1. **Identify functional groups** and reactive sites
2. **Consider electronic factors**:
   - Electron-rich sites act as nucleophiles
   - Electron-poor sites act as electrophiles
3. **Evaluate steric factors**:
   - Bulky groups hinder approach of reagents
   - Strain can drive reactivity
4. **Consider solvent effects**:
   - Polar protic solvents stabilize charged species
   - Aprotic polar solvents enhance nucleophilicity
5. **Apply conceptual tools**:
   - Resonance structures
   - Inductive effects
   - Hyperconjugation
   - Aromaticity

## Mechanistic Investigations

### Experimental Methods

1. **Kinetic studies**: Determine rate laws and order of reaction
2. **Isotopic labeling**: Track atoms through reaction pathway
3. **Stereochemical analysis**: Observe stereochemical outcomes
4. **Crossover experiments**: Detect intermediates
5. **Solvent effects**: Probe involvement of charged species
6. **Spectroscopic methods**: Detect/characterize intermediates

### Computational Approaches

1. **Transition state modeling**: Locate energy barriers
2. **Reaction coordinate calculations**: Map complete reaction pathways
3. **Molecular orbital analysis**: Evaluate electronic factors

## Advanced Mechanistic Concepts

### Neighboring Group Participation

```
       X
       |
R-C-C-LG → R-C-C → R-C-C-Nu
   |        \|/       |
   Y         Y        Y
```

- **Characteristics**: Internal nucleophilic assistance
- **Effect**: Rate enhancement, stereochemical control
- **Examples**: Anchimeric assistance in epoxide formation

### Stereoelectronic Effects

- **Characteristics**: Electronic effects dependent on spatial orientation
- **Examples**: Anomeric effect, antiperiplanar alignment in E2

### Enzyme-like Catalysis

- **Characteristics**: Approximation, orientation, strain, nucleophilic/electrophilic catalysis
- **Examples**: Phase-transfer catalysis, organocatalysis

---

**References**:
1. Carey, F. A., & Sundberg, R. J. (2007). Advanced Organic Chemistry, Part A: Structure and Mechanisms (5th ed.). Springer.
2. Clayden, J., Greeves, N., & Warren, S. (2012). Organic Chemistry (2nd ed.). Oxford University Press.
3. Anslyn, E. V., & Dougherty, D. A. (2006). Modern Physical Organic Chemistry. University Science Books. 