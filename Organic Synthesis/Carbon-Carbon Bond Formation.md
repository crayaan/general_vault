---
aliases:
  - C-C Bond Formation
  - Carbon Bond Formation
---

# Carbon-Carbon Bond Formation

Carbon-carbon bond formation reactions are the cornerstone of organic synthesis, allowing for the construction of complex molecular frameworks from simpler building blocks. These reactions are essential for increasing molecular complexity and building carbon skeletons of target molecules.

## Classical Methods

### 1. Grignard Reactions

Organomagnesium halides (RMgX) react with carbonyl compounds to form new C-C bonds.

```
R-MgX + R'₂C=O → R-C(OH)R'₂
```

- **Scope**: Aldehydes, ketones, esters, acid chlorides, CO₂
- **Limitations**: Incompatible with acidic protons, moisture-sensitive
- **Stereoselectivity**: Poor control, bulky groups favor equatorial attack
- **Key variations**: 
  - Alkyl/aryl/vinyl/alkynyl Grignard reagents
  - Reformatsky reaction (α-halo esters)

### 2. Organolithium Reagents

Similar to Grignard reagents but more reactive and less selective.

```
R-Li + R'₂C=O → R-C(OLi)R'₂ → R-C(OH)R'₂
```

- **Advantages**: Higher reactivity than Grignard reagents
- **Limitations**: Extreme moisture sensitivity, side reactions
- **Strategic uses**: Lithium-halogen exchange, directed ortho-metallation

### 3. Alkylation of Enolates

Deprotonation of carbonyls creates nucleophilic enolates that attack electrophiles.

```
      O             O                        O
      ‖             ‖                        ‖
R-CH₂-C-R' → R-CH-C-R' → R-CH(R")-C-R'
                |
                R"X
```

- **Base choice**: LDA (kinetic), NaOEt (thermodynamic)
- **Electrophiles**: Alkyl halides, Michael acceptors
- **Key variations**:
  - Malonic ester synthesis
  - Acetoacetic ester synthesis
  - Alkylation of hydrazones (Stork)

### 4. Aldol Reaction

Enolates react with carbonyls to form β-hydroxy carbonyl compounds.

```
      O            O        OH O
      ‖            ‖        |  ‖
R-CH₂-C-R' + R"CHO → R-CH-CH-C-R'
                         |
                         R"
```

- **Variations**:
  - Directed aldol (lithium enolates)
  - Crossed aldol (mixed aldehydes)
  - Mukaiyama aldol (silyl enol ethers)
  - Evans aldol (chiral auxiliaries)
- **Stereochemistry**: Can be controlled via chiral auxiliaries or catalysts

### 5. Claisen Condensation

Ester enolates react with esters to form β-keto esters.

```
      O             O           O   O
      ‖             ‖           ‖   ‖
R-CH₂-C-OR' + R"-C-OR' → R-CH₂-C-CH-C-OR'
                                   |
                                  R"
```

- **Variations**:
  - Dieckmann condensation (intramolecular)
  - Mixed Claisen (two different esters)
  - Schmidt modification (acid chlorides)

## Transition Metal-Catalyzed Methods

### 1. Palladium-Catalyzed Cross-Coupling

#### Suzuki-Miyaura Coupling

Couples organoboron compounds with organohalides.

```
R-B(OH)₂ + R'-X → [Pd] → R-R' + X-B(OH)₂
```

- **Catalyst**: Pd(PPh₃)₄, Pd(dppf)Cl₂, etc.
- **Base**: K₂CO₃, Cs₂CO₃, K₃PO₄
- **Advantages**: Mild conditions, tolerates functional groups, air/moisture stable
- **R groups**: Aryl, vinyl, alkyl (challenging)

#### Stille Coupling

Couples organostannanes with organohalides.

```
R-SnBu₃ + R'-X → [Pd] → R-R' + X-SnBu₃
```

- **Catalyst**: Pd(PPh₃)₄, Pd₂(dba)₃/P(o-tol)₃
- **Advantages**: No base required, broad functional group tolerance
- **Limitations**: Toxicity of tin compounds

#### Negishi Coupling

Couples organozinc compounds with organohalides.

```
R-ZnX + R'-X → [Pd] → R-R' + ZnX₂
```

- **Catalyst**: Pd(PPh₃)₄, Pd(dppf)Cl₂
- **Advantages**: Higher reactivity than Suzuki
- **Limitations**: Moisture sensitivity

#### Kumada Coupling

Couples Grignard reagents with organohalides.

```
R-MgX + R'-X → [Pd] or [Ni] → R-R' + MgX₂
```

- **Catalyst**: PdCl₂(dppf), Ni(dppp)Cl₂
- **Advantages**: Direct use of Grignard reagents
- **Limitations**: Poor functional group tolerance

#### Heck Reaction

Couples aryl/vinyl halides with alkenes.

```
R-X + CH₂=CHR' → [Pd] → R-CH=CH-R' + H-X
```

- **Catalyst**: Pd(OAc)₂/PPh₃
- **Base**: NEt₃, K₂CO₃
- **Regioselectivity**: Favors β-arylation
- **Stereoselectivity**: Usually trans

#### Sonogashira Coupling

Couples terminal alkynes with aryl/vinyl halides.

```
R-C≡CH + R'-X → [Pd]/[Cu] → R-C≡C-R' + H-X
```

- **Catalyst**: Pd(PPh₃)₂Cl₂/CuI
- **Base**: NEt₃, piperidine
- **Advantages**: Mild conditions

### 2. Olefin Metathesis

Redistribution of carbon-carbon double bonds using ruthenium or molybdenum catalysts.

```
R-CH=CH₂ + R'-CH=CH₂ → R-CH=CH-R' + CH₂=CH₂
```

- **Catalysts**: Grubbs (Ru), Schrock (Mo)
- **Variations**:
  - Ring-closing metathesis (RCM)
  - Ring-opening metathesis polymerization (ROMP)
  - Cross metathesis (CM)
  - Enyne metathesis
- **Stereoselectivity**: Depends on catalyst and substrates

### 3. C-H Activation

Direct functionalization of C-H bonds without pre-activation.

```
R-H + X-R' → [M] → R-R' + H-X
```

- **Catalysts**: Palladium, ruthenium, rhodium, iridium
- **Directing groups**: Pyridine, amides, carboxylic acids
- **Advantages**: Step economy, atom efficiency
- **Challenges**: Regioselectivity, requires directing groups

## Pericyclic Reactions

### 1. Diels-Alder Reaction

[4+2] Cycloaddition between a diene and dienophile.

```
      \      /
       \    /
        \  /
         \/
```

- **Stereochemistry**: Endo kinetic product, exo thermodynamic product
- **Regioselectivity**: Electron-withdrawing groups direct addition
- **Variations**:
  - Hetero-Diels-Alder
  - Intramolecular Diels-Alder
  - Inverse electron demand

### 2. [3+2] Cycloadditions

Cycloaddition between 1,3-dipoles and dipolarophiles.

```
R-N=N=N + C=C → five-membered heterocycle
```

- **1,3-Dipoles**: Azides, nitrones, diazo compounds
- **Applications**: Heterocycle synthesis, click chemistry
- **Example**: Huisgen azide-alkyne cycloaddition

### 3. Claisen Rearrangement

[3,3]-Sigmatropic rearrangement of allyl vinyl ethers.

```
        O
       / \
      /   \
     /     \
    →       →
```

- **Variations**:
  - Ireland-Claisen (ester enolates)
  - Johnson-Claisen (orthoester)
  - Eschenmoser-Claisen (amide acetals)
- **Stereoselectivity**: Chair-like transition state

## Umpolung (Polarity Reversal) Methods

### 1. Acyl Anion Equivalents

#### Corey-Seebach Dithiane Method

```
       S—S
      /   \
     /     \
    CH      + R-X → [alkylation] → [hydrolysis] → R-CHO
```

- **Advantages**: Reverses normal carbonyl reactivity
- **Limitations**: Multiple steps, use of toxic mercury reagents

#### Benzoin Condensation

```
2 Ar-CHO → [cyanide or NHC] → Ar-CH(OH)-CO-Ar
```

- **Catalysts**: Cyanide ion, N-heterocyclic carbenes
- **Variations**: Stetter reaction (addition to Michael acceptors)

## Radical Methods

### 1. Atom Transfer Radical Addition

```
R-X + C=C → [radical initiator] → R-C-C-X
```

- **Initiators**: AIBN, peroxides, light
- **Common reagents**: Bu₃SnH, (Me₃Si)₃SiH
- **Applications**: Cyclization reactions, cascade reactions
- **Example**: Barton-McCombie deoxygenation

### 2. Giese Reaction

Radical addition to electron-deficient alkenes.

```
R· + CH₂=CH-EWG → R-CH₂-CH·-EWG
```

- **R sources**: Alkyl halides, alcohols (via Barton esters)
- **EWG**: Carbonyl, nitrile, sulfone, etc.

## Organocatalytic Methods

### 1. Enamine Catalysis

```
       R                   R
       |                   |
R'₂NH + O=C → R'₂N-C= → R'₂N-C-R" → O=C-R"
       |                |            |
       H                H            H
```

- **Catalysts**: Proline, imidazolidinones
- **Reactions**: Aldol, Mannich, Michael additions
- **Advantages**: Mild conditions, often enantioselective

### 2. Iminium Catalysis

```
R'₂NH + O=C-CH= → R'₂N=C-CH= + nucleophile → R'₂N=C-CH-Nu
```

- **Catalysts**: Imidazolidinones, diarylprolinol ethers
- **Reactions**: Michael additions, cycloadditions
- **Stereoselectivity**: Controlled by catalyst structure

## Strategic Considerations

### 1. Retrosynthetic Analysis

- **Functional group interconversions**: Identify key functional groups
- **Disconnection strategies**: Break at C-C bonds
- **Synthetic equivalents**: Match polarity needs

### 2. Protecting Group Strategy

- **Chemoselectivity**: Protect reactive groups
- **Orthogonal protection**: Compatible deprotection conditions
- **Minimal protection**: Reduce step count

### 3. Step Economy

- **Cascade reactions**: Multiple C-C bonds in one pot
- **Tandem catalysis**: Multiple catalytic cycles
- **Multicomponent reactions**: Three or more components

## Comparison Table of Key C-C Bond Forming Methods

| Method | Scope | Functional Group Tolerance | Stereoselectivity | Green Chemistry Factors |
|--------|-------|----------------------------|-------------------|-----------------------|
| **Grignard** | Carbonyls | Poor | Low | Poor atom economy |
| **Aldol** | Aldehydes/ketones | Moderate | High with auxiliaries | Moderate efficiency |
| **Suzuki** | Aryl, vinyl | Excellent | N/A for Csp²-Csp² | Moderate to good |
| **Diels-Alder** | Conjugated systems | Good | High | Excellent atom economy |
| **Metathesis** | Alkenes | Excellent | Moderate | Catalytic |
| **C-H Activation** | C-H bonds | Variable | Variable | High atom economy |
| **Organocatalysis** | Various | Good | Often excellent | Environmentally friendly |

## Contemporary Developments

1. **Photoredox catalysis**: Light-driven C-C bond formation
2. **Dual catalysis**: Combining transition metal and organocatalysis
3. **Flow chemistry**: Continuous process C-C bond formation
4. **Electrochemical methods**: Electron-transfer-driven reactions
5. **Artificial intelligence**: Predicting optimal C-C bond forming conditions

---

**References**:
1. Kürti, L., & Czakó, B. (2005). Strategic Applications of Named Reactions in Organic Synthesis. Elsevier Academic Press.
2. Nicolaou, K. C., & Sorensen, E. J. (1996). Classics in Total Synthesis. Wiley-VCH.
3. Hartwig, J. F. (2010). Organotransition Metal Chemistry: From Bonding to Catalysis. University Science Books. 