---
aliases:
  - Alkene Transformations
  - C=C Reactions
---

# Alkene Reactions

Alkenes (compounds containing C=C double bonds) are among the most versatile functional groups in organic chemistry due to their rich reaction chemistry. The π bond serves as an excellent site for various transformations, allowing for the synthesis of numerous other functional groups.

## Categorizing Alkene Reactions

Alkene reactions can be organized into three main mechanistic categories:

1. **Reactions involving carbocation intermediates**
2. **Reactions involving three-membered ring intermediates**
3. **Concerted reactions (one-step processes)**

This organization helps in predicting both regioselectivity and stereochemistry of additions to alkenes.

## Common Reaction Types

### 1. Electrophilic Additions

The general pattern: an electrophile attacks the π bond, generating a carbocation intermediate which is subsequently captured by a nucleophile.

| Reaction | Reagents | Products | Stereochemistry | Regioselectivity | Memory Aid |
|----------|----------|----------|----------------|-----------------|------------|
| **Hydrohalogenation** | HX (X = Cl, Br, I) | Alkyl halides | Random | Markovnikov | "H goes where H's are" |
| **Acid-catalyzed Hydration** | H₂O, H⁺ | Alcohols | Random | Markovnikov | "Water needs acid to add" |
| **Oxymercuration-Demercuration** | Hg(OAc)₂, H₂O then NaBH₄ | Alcohols | Random | Markovnikov | "Mercury helps water add Markovnikov" |

#### Mechanism Example: HBr Addition

```
R₂C=CR₂ + H-Br → R₂C⁺-CR₂H + Br⁻ → R₂CBr-CR₂H
```

* **Regioselectivity**: H⁺ adds to the less substituted carbon (forming more stable carbocation)
* **Key intermediate**: Carbocation (planar, sp² hybridized)
* **Memory aid**: "The rich get richer" (more substituted carbon gets the partial positive charge)

### 2. Three-Membered Ring Intermediates

These reactions proceed through cyclic intermediates, resulting in trans addition.

| Reaction | Reagents | Products | Stereochemistry | Regioselectivity | Memory Aid |
|----------|----------|----------|----------------|-----------------|------------|
| **Halogenation** | X₂ (X = Cl, Br) | Dihalides | Trans | - | "Bromine adds trans (see the X bridge)" |
| **Halohydrin Formation** | X₂, H₂O | Halohydrins | Trans | Markovnikov | "X⁺ goes to make the more stable cation" |
| **Epoxidation** | RCO₃H (peroxyacids) | Epoxides | Syn | - | "Peroxyacids make epoxides" |
| **Acid-catalyzed Epoxide Opening** | H⁺, Nu⁻ | trans-1,2-difunctionalized | Trans | Markovnikov | "Acid opens the more substituted side" |

#### Mechanism Example: Br₂ Addition

```
       Br
       |
R₂C=CR₂ + Br₂ → R₂C—CR₂ → R₂CBr-CR₂Br
       |
       Br⁺
```

* **Key intermediate**: Bromonium ion (cyclic, bridged structure)
* **Memory aid**: "The bromonium bridge forces backside attack"

### 3. Concerted (One-Step) Reactions

These reactions occur in a single step without discrete intermediates.

| Reaction | Reagents | Products | Stereochemistry | Regioselectivity | Memory Aid |
|----------|----------|----------|----------------|-----------------|------------|
| **Hydroboration-Oxidation** | BH₃·THF then H₂O₂, OH⁻ | Alcohols | Syn | Anti-Markovnikov | "BH₃ is bulky and boron is electron-deficient" |
| **Catalytic Hydrogenation** | H₂, Pt/Pd/Ni | Alkanes | Syn | - | "H₂ adds from the less hindered side" |
| **Dihydroxylation** | OsO₄ or cold KMnO₄ | Diols (vicinal) | Syn | - | "Os/Mn deliver both OH groups together" |
| **Ozonolysis** | 1. O₃ 2. Zn/H₂O or Me₂S | Aldehydes/Ketones | - | - | "O3 breaks the double bond" |

#### Mechanism Example: OsO₄ Dihydroxylation

```
       O       O
       ‖       ‖
R₂C=CR₂ + O=Os=O → R₂C—CR₂ → R₂C(OH)-C(OH)R₂
       |       |
       O———Os—O
```

* **Key features**: Concerted addition through a cyclic transition state
* **Memory aid**: "Os helps oxygens to add simultaneously from the same side"

## Reaction Selection Flowchart

To remember when to use specific reagents for desired transformations:

1. **Want an alcohol?**
   * Markovnikov: H₂O/H⁺ or Hg(OAc)₂/H₂O then NaBH₄
   * Anti-Markovnikov: BH₃·THF then H₂O₂/OH⁻

2. **Want a halide?**
   * Markovnikov: HX (X = Cl, Br, I)
   * Anti-Markovnikov: HBr/ROOR (radical conditions)
   * Dihalide: X₂ (X = Cl, Br)

3. **Want a diol?**
   * Syn: OsO₄ or cold KMnO₄
   * Anti: Epoxidation followed by base-catalyzed ring opening

4. **Want carbonyl compounds?**
   * Aldehydes/Ketones: Ozonolysis
   * Carboxylic acids: Hot KMnO₄

## Stereochemistry at a Glance

| Reaction Type | Intermediate | Overall Stereochemistry |
|---------------|--------------|-------------------------|
| Carbocation | Planar | Random |
| Three-membered ring | Cyclic | Trans |
| Concerted | None | Syn |

## Mnemonic Devices

1. **"AEM"** - For types of addition: **A**nti-Markovnikov, **E**lectrophilic, **M**arkovnikov
2. **"SHHHB"** - Anti-Markovnikov reagents: **S**ulfuric acid + HgSO₄, **H**ydrogen + catalyst, **H**ydroboration, **H**ydroperoxides, **B**ulgier borane
3. **"HHOBr"** - Markovnikov reagents: **H**ydrogen halides, **H**ydration, **O**xymercuration, **Br**₂ + H₂O

## Multi-Step Synthesis Applications

Alkene reactions are frequently used in multi-step synthesis:

1. **Two-step alcohol synthesis**:
   ```
   Alkene → Epoxide → Alcohol (with stereocontrol)
   ```

2. **Vicinal diol synthesis**:
   ```
   Alkene → Diol (syn via OsO₄ or anti via epoxide/opening)
   ```

3. **Carbon chain cleavage**:
   ```
   Alkene → Ozonide → Aldehydes/Ketones
   ```

4. **Functional group interconversion**:
   ```
   Alkene → Halide → Grignard reagent → various products
   ```

---

**References**:
1. Clayden, J., Greeves, N., & Warren, S. (2012). Organic Chemistry (2nd ed.). Oxford University Press.
2. Organic Chemistry Tutor (2023). Alkenes Cheat Sheet and Summary Notes. 