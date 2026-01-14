import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

# Extended mechanism rules including additional pathways.
mechanism_rules = {
    "HACA": {
        "reaction_smarts": "[cH:1].[C]#[C]>>[c:1][C]#[C]",
        "reagents": "C#C"
    },
    "DA": {
        "reaction_smarts": "[c:1][c:2]>>[c:1]1[c:2]cc1",
        "reagents": "C#C"
    },
    "HAVA": {
        "reaction_smarts": "[cH:1].[C]=C[C]#[C]>>[c:1]C=CC#C",
        "reagents": "C=CC#C"
    },
    "MAC": {
        "reaction_smarts": "[cH:1].[CH3]>>[c:1]C",
        "reagents": "[CH3]"
    },
    "EAM": {
        "reaction_smarts": "[cH:1].[C]#[C]>>[c:1]C#C",
        "reagents": "C#C"
    },
    "N2": {
        "reaction_smarts": "[C:1]c1ccccc1.C=CC#C>>[C:1]c1ccc(C=CC#C)cc1",
        "reagents": "C=CC#C"
    },
    "N3": {
        "reaction_smarts": "[c:1]1cccc1.[c:2]1cccc1>>C1=CC2=CC=CC=C2C1",
        "reagents": ""
    },
    "N4": {
        "reaction_smarts": "[C#C:1][C:2].[C:3]c1ccccc1>>[C:3]c1ccccc1C#C",
        "reagents": ""
    },
    "N5": {
        "reaction_smarts": "[C:1]c1ccccc1.C=CC=C>>[C:1]c1ccc(C=CC=C)cc1",
        "reagents": "C=CC=C"
    },
    "N6": {
        "reaction_smarts": "[c:1]1ccc2c(c1)C=C2.[CH3]>>C1=CC2=CC=CC=C2C1",
        "reagents": "[CH3]"
    },
    "PA_allene": {
        "reaction_smarts": "[C:1]c1ccccc1.C=C=C>>[C:1]c1ccc(C=C=C)cc1",
        "reagents": "C=C=C"
    },
    "PA_propyne": {
        "reaction_smarts": "[C:1]c1ccccc1.CC#C>>[C:1]c1ccc(CC#C)cc1",
        "reagents": "CC#C"
    },
    "I2": {
        "reaction_smarts": "[C#C:1]C.[cH:2]1ccccc1>>[c:2]1ccccc1C#C",
        "reagents": "C#CC"
    },
    "I3": {
        "reaction_smarts": "[C:1]c1ccccc1.C#C>>[C:1]c1ccc(C#C)cc1",
        "reagents": "C#C"
    },
    "I4": {
        "reaction_smarts": "[C:1]c1ccccc1.CC=C>>[C:1]c1ccc(CC=C)cc1",
        "reagents": "CC=C"
    },
    "I4_allyl": {
        "reaction_smarts": "[C:1]c1ccccc1.[C]C=C>>[C:1]c1ccc(CC=C)cc1",
        "reagents": "[C]C=C"
    }
}

# Prepare a table summarizing the reaction mechanisms.
mechanism_data = [
    {"Mechanism ID": "HACA", "Reaction Name": "Hydrogen Abstraction–Acetylene Addition",
     "Description": "Abstracts a hydrogen from an aromatic site and adds acetylene.", 
     "Reaction SMARTS": "[cH:1].[C]#[C]>>[c:1][C]#[C]", "Reagents": "C#C"},
    {"Mechanism ID": "DA", "Reaction Name": "Diels–Alder Mechanism (placeholder)",
     "Description": "Cycloaddition forming a cyclic product.", 
     "Reaction SMARTS": "[c:1][c:2]>>[c:1]1[c:2]cc1", "Reagents": "C#C"},
    {"Mechanism ID": "HAVA", "Reaction Name": "Hydrogen Abstraction–Vinyl Acetylene Addition",
     "Description": "Uses vinyl acetylene for vinylation of an aromatic radical.", 
     "Reaction SMARTS": "[cH:1].[C]=C[C]#[C]>>[c:1]C=CC#C", "Reagents": "C=CC#C"},
    {"Mechanism ID": "MAC", "Reaction Name": "Methyl Addition and Cyclization",
     "Description": "Adds a methyl group to an aromatic radical.", 
     "Reaction SMARTS": "[cH:1].[CH3]>>[c:1]C", "Reagents": "[CH3]"},
    {"Mechanism ID": "EAM", "Reaction Name": "Ethynyl Addition Mechanism",
     "Description": "Adds an ethynyl group to an aromatic radical.", 
     "Reaction SMARTS": "[cH:1].[C]#[C]>>[c:1]C#C", "Reagents": "C#C"},
    {"Mechanism ID": "N2", "Reaction Name": "Vinylacetylene Addition to Phenyl Radical",
     "Description": "Adds vinylacetylene to a phenyl radical.", 
     "Reaction SMARTS": "[C:1]c1ccccc1.C=CC#C>>[C:1]c1ccc(C=CC#C)cc1", "Reagents": "C=CC#C"},
    {"Mechanism ID": "N3", "Reaction Name": "Cyclopentadienyl Radical Recombination",
     "Description": "Recombines two cyclopentadienyl radicals to form a naphthalene-like structure.", 
     "Reaction SMARTS": "[c:1]1cccc1.[c:2]1cccc1>>C1=CC2=CC=CC=C2C1", "Reagents": ""},
    {"Mechanism ID": "N4", "Reaction Name": "Propargyl + Benzyl Radical Reaction",
     "Description": "Couples a propargyl radical with a benzyl radical.", 
     "Reaction SMARTS": "[C#C:1][C:2].[C:3]c1ccccc1>>[C:3]c1ccccc1C#C", "Reagents": ""},
    {"Mechanism ID": "N5", "Reaction Name": "1,3-Butadiene Addition to Phenyl Radical",
     "Description": "Adds 1,3-butadiene to a phenyl radical.", 
     "Reaction SMARTS": "[C:1]c1ccccc1.C=CC=C>>[C:1]c1ccc(C=CC=C)cc1", "Reagents": "C=CC=C"},
    {"Mechanism ID": "N6", "Reaction Name": "Indene/Indenyl to Naphthalene via Methylation",
     "Description": "Methylates an indene/indenyl radical and rearranges to naphthalene.", 
     "Reaction SMARTS": "[c:1]1ccc2c(c1)C=C2.[CH3]>>C1=CC2=CC=CC=C2C1", "Reagents": "[CH3]"},
    {"Mechanism ID": "PA_allene", "Reaction Name": "Phenyl + Allene Reaction",
     "Description": "Adds an allene to a phenyl radical.", 
     "Reaction SMARTS": "[C:1]c1ccccc1.C=C=C>>[C:1]c1ccc(C=C=C)cc1", "Reagents": "C=C=C"},
    {"Mechanism ID": "PA_propyne", "Reaction Name": "Phenyl + Propyne Reaction",
     "Description": "Adds a propyne to a phenyl radical.", 
     "Reaction SMARTS": "[C:1]c1ccccc1.CC#C>>[C:1]c1ccc(CC#C)cc1", "Reagents": "CC#C"},
    {"Mechanism ID": "I2", "Reaction Name": "Propargyl with Benzene Reaction",
     "Description": "Couples a propargyl radical with benzene/phenyl.", 
     "Reaction SMARTS": "[C#C:1]C.[cH:2]1ccccc1>>[c:2]1ccccc1C#C", "Reagents": "C#CC"},
    {"Mechanism ID": "I3", "Reaction Name": "Benzyl with Acetylene Reaction",
     "Description": "Acetylene adds to a benzyl radical.", 
     "Reaction SMARTS": "[C:1]c1ccccc1.C#C>>[C:1]c1ccc(C#C)cc1", "Reagents": "C#C"},
    {"Mechanism ID": "I4", "Reaction Name": "Phenyl with Propene Reaction",
     "Description": "Propene reacts with a phenyl radical.", 
     "Reaction SMARTS": "[C:1]c1ccccc1.CC=C>>[C:1]c1ccc(CC=C)cc1", "Reagents": "CC=C"},
    {"Mechanism ID": "I4_allyl", "Reaction Name": "Phenyl with Allyl Radical Reaction",
     "Description": "An allyl radical couples with a phenyl radical.", 
     "Reaction SMARTS": "[C:1]c1ccccc1.[C]C=C>>[C:1]c1ccc(CC=C)cc1", "Reagents": "[C]C=C"}
]

mechanism_df = pd.DataFrame(mechanism_data)

# --- Streamlit UI ---

st.title("Multi-Step PAH Formation Simulator – Extended Mechanisms")

# Display the Reaction Mechanisms Summary Table.
st.header("Reaction Mechanisms Summary")
st.write("Below is a summary of the reaction mechanisms available in this application:")
st.dataframe(mechanism_df)

st.write("""
This application simulates multi-step PAH formation. Enter an initial precursor SMILES, select a reaction mechanism, 
and click "Apply Reaction Step" to perform a reaction. The product from each step becomes the reactant for the next step.
Use "Reset Reaction" to start over.
""")

# Initialize or reset the intermediate product in session state.
if "intermediate_smiles" not in st.session_state:
    st.session_state.intermediate_smiles = None

if st.button("Reset Reaction"):
    st.session_state.intermediate_smiles = None

# If no intermediate exists, ask for an initial reactant.
if st.session_state.intermediate_smiles is None:
    initial_smiles = st.text_input("Enter initial reactant SMILES:", "Brc1ccccc1")
else:
    st.write("**Current Intermediate:**", st.session_state.intermediate_smiles)
    current_mol = Chem.MolFromSmiles(st.session_state.intermediate_smiles)
    if current_mol:
        st.image(Draw.MolToImage(current_mol, size=(300, 300)), caption="Current Intermediate Structure")
    initial_smiles = st.session_state.intermediate_smiles

# Select the reaction mechanism to apply in the next step.
mechanism = st.selectbox("Select Reaction Mechanism for Next Step", options=list(mechanism_rules.keys()))

def apply_mechanism(precursor_smiles, mechanism):
    if mechanism not in mechanism_rules:
        st.error("Mechanism not implemented.")
        return None

    rule = mechanism_rules[mechanism]
    rxn_smarts = rule["reaction_smarts"]
    reagents_str = rule["reagents"]

    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    except Exception as e:
        st.error(f"Error creating reaction from SMARTS: {rxn_smarts}\n{e}")
        return None

    precursor_mol = Chem.MolFromSmiles(precursor_smiles)
    if precursor_mol is None:
        st.error(f"Invalid precursor SMILES: {precursor_smiles}")
        return None

    reagent_mols = []
    if reagents_str:
        # Allow multiple reagents separated by commas.
        for smi in reagents_str.split(","):
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                st.error(f"Invalid reagent SMILES: {smi}")
                return None
            reagent_mols.append(mol)

    reactants = (precursor_mol,) + tuple(reagent_mols)
    try:
        product_sets = rxn.RunReactants(reactants)
    except Exception as e:
        st.error(f"Error during reaction run: {e}")
        return None

    # For simplicity, choose the first product from the first reaction set.
    if product_sets and len(product_sets) > 0:
        product = product_sets[0][0]
        try:
            Chem.SanitizeMol(product)
            return Chem.MolToSmiles(product)
        except Exception as e:
            st.error(f"Error sanitizing product: {e}")
            return None
    else:
        st.error("No product generated. Check the reaction SMARTS.")
        return None

# Button to apply the selected reaction step.
if st.button("Apply Reaction Step"):
    new_product = apply_mechanism(initial_smiles, mechanism)
    if new_product:
        st.session_state.intermediate_smiles = new_product
        st.success(f"Reaction succeeded! New Intermediate: {new_product}")
        new_mol = Chem.MolFromSmiles(new_product)
        if new_mol:
            st.image(Draw.MolToImage(new_mol, size=(300, 300)), caption="New Intermediate Structure")
    else:
        st.error("Reaction failed. Try a different mechanism or check your input.")
