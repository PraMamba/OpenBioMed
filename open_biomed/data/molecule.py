from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

import copy
from datetime import datetime
import gzip
import math
import numpy as np
import os
import pickle
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors, Lipinski
from rdkit.Chem.AllChem import RWMol
from rdkit.six import iteritems
from rdkit.six.moves import cPickle
import re

from open_biomed.core.tool import Tool
from open_biomed.data.text import Text
from open_biomed.utils.exception import MoleculeConstructError

_fscores = None

class Molecule:
    def __init__(self) -> None:
        super().__init__()
        self.name = None

        # basic properties: 1D SMILES/SELFIES strings, RDKit mol object, 2D graphs, 3D coordinates
        self.smiles = None
        self.selfies = None
        self.rdmol = None
        self.graph = None
        self.conformer = None

        # other related properties: image, textual descriptions and identifier in knowledge graph
        self.img = None
        self.description = None
        self.kg_accession = None

    @classmethod
    def from_smiles(cls, smiles: str) -> Self:
        # initialize a molecule with a SMILES string
        molecule = cls()
        molecule.smiles = smiles
        # molecule._add_rdmol(base="smiles")
        return molecule

    @classmethod
    def from_selfies(cls, selfies: str) -> Self:
        import selfies as sf
        molecule = cls()
        molecule.selfies = selfies
        molecule.smiles = sf.decoder(selfies)
        return molecule

    @classmethod
    def from_rdmol(cls, rdmol: RWMol) -> Self:
        # initialize a molecule with a RDKit molecule
        molecule = cls()
        molecule.rdmol = rdmol
        molecule.smiles = Chem.MolToSmiles(rdmol)
        conformer = rdmol.GetConformer()
        if conformer is not None:
            molecule.conformer = np.array(conformer.GetPositions())
        return molecule

    @classmethod
    def from_pdb_file(cls, pdb_file: str) -> Self:
        # initialize a molecule with a pdb file
        pass

    @classmethod
    def from_sdf_file(cls, sdf_file: str) -> Self:
        # initialize a molecule with a sdf file
        loader = Chem.SDMolSupplier(sdf_file)
        for mol in loader:
            if mol is not None:
                molecule = Molecule.from_rdmol(mol)
                conformer = mol.GetConformer()
                molecule.conformer = np.array(conformer.GetPositions())
        molecule.name = sdf_file.split("/")[-1].strip(".sdf")
        return molecule

    @classmethod
    def from_image_file(cls, image_file: str) -> Self:
        # initialize a molecule with a image file
        pass

    @classmethod
    def from_binary_file(cls, file: str) -> Self:
        return pickle.load(open(file, "rb"))

    @staticmethod
    def convert_smiles_to_rdmol(smiles: str, canonical: bool=True) -> RWMol:
        # Convert the smiles string into rdkit mol
        # If the smiles is invalid, raise MolConstructError
        pass

    @staticmethod
    def generate_conformer(rdmol: RWMol, method: str='mmff', num_conformers: int=1) -> np.ndarray:
        # Generate 3D conformer with algorithms in RDKit
        # TODO: identify if ML-based conformer generation can be applied
        pass

    def _add_name(self) -> None:
        if self.name is None:
            self.name = "mol_" + re.sub(r"[-:.]", "_", datetime.now().isoformat(sep="_", timespec="milliseconds"))

    def _add_smiles(self, base: str='rdmol') -> None:
        # Add class property: smiles, based on selfies / rdmol / graph, default: rdmol
        pass

    def _add_selfies(self, base: str='smiles') -> None:
        import selfies as sf
        # Add class property: selfies, based on smiles / selfies / rdmol / graph, default: smiles
        if base == "smiles":
            self.selfies = sf.encoder(self.smiles, strict=False)
        else:
            raise NotImplementedError

    def _add_rdmol(self, base: str='smiles') -> None:
        # Add class property: rdmol, based on smiles / selfies / graph, default: smiles
        if self.rdmol is not None:
            return
        if base == 'smiles':
            self.rdmol = Chem.MolFromSmiles(self.smiles)
        if self.conformer is not None:
            conf = mol_array_to_conformer(self.conformer)
            self.rdmol.AddConformer(conf)

    def _add_conformer(self, mode: str='2D', base: str='rdmol') -> None:
        # Add class property: conformer, based on smiles / selfies / rdmol, default: rdmol
        if self.conformer is None:
            self._add_rdmol()
            if mode == '2D':
                AllChem.Compute2DCoords(self.rdmol)
            elif mode == '3D':
                self.rdmol = Chem.AddHs(self.rdmol)
                AllChem.EmbedMolecule(self.rdmol)
                AllChem.MMFFOptimizeMolecule(self.rdmol)
            conformer = self.rdmol.GetConformer()
            self.conformer = np.array(conformer.GetPositions())
    
    def _add_description(self, text_database: Dict[Any, Text], identifier_key: str='SMILES', base: str='smiles') -> None:
        # Add class property: description, based on smiles / selfies / rdmol, default: smiles
        pass

    def _add_kg_accession(self, kg_database: Dict[Any, str], identifier_key: str='SMILES', base: str='smiles') -> None:
        # Add class property: kg_accession, based on smiles / selfies / rdmol, default: smiles
        pass

    def save_sdf(self, file: Optional[str]=None, overwrite: bool=False) -> str:
        if file is None:
            self._add_name()
            file = f"./tmp/{self.name}.sdf"

        if not os.path.exists(file) or overwrite:
            writer = Chem.SDWriter(file)
            self._add_rdmol()
            self._add_conformer()
            writer.write(self.rdmol)
        return file

    def save_binary(self, file: Optional[str]=None, overwrite: bool=False) -> str:
        if file is None:
            self._add_name()
            file = f"./tmp/{self.name}.pkl"

        if not os.path.exists(file) or overwrite:
            pickle.dump(self, open(file, "wb"))
        return file

    def get_num_atoms(self) -> None:
        self._add_rdmol()
        return self.rdmol.GetNumAtoms()
    
    def calc_qed(self) -> float:
        try:
            from rdkit.Chem.QED import qed
            self._add_rdmol()
            return qed(self.rdmol)
        except Exception:
            return 0.0

    def calc_sa(self) -> float:
        self._add_rdmol()
        sa = calc_sa_score(self.rdmol)
        sa_norm = round((10 - sa) / 9, 2)
        return sa_norm

    def calc_logp(self) -> float:
        from rdkit.Chem.Crippen import MolLogP
        self._add_rdmol()
        return MolLogP(self.rdmol)

    def calc_lipinski(self) -> float:
        try:
            self._add_rdmol()
            mol = copy.deepcopy(self.rdmol)
            Chem.SanitizeMol(mol)
            rule_1 = Descriptors.ExactMolWt(mol) < 500
            rule_2 = Lipinski.NumHDonors(mol) <= 5
            rule_3 = Lipinski.NumHAcceptors(mol) <= 10
            logp = self.calc_logp()
            rule_4 = (logp >= -2) & (logp <= 5)
            rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
            return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
        except Exception:
            return 0.0

    def calc_distance(self) -> float:
        self._add_conformer()
        pdist = self.conformer[None, :] - self.conformer[:, None]
        return np.sqrt(np.sum(pdist ** 2, axis=-1))

    def __str__(self) -> str:
        return self.smiles

def molecule_fingerprint_similarity(mol1: Molecule, mol2: Molecule, fingerprint_type: str="morgan") -> float:
    # Calculate the fingerprint similarity of two molecules
    try:
        mol1._add_rdmol()
        mol2._add_rdmol()
        if fingerprint_type == "morgan":
            fp1 = AllChem.GetMorganFingerprint(mol1.rdmol, 2)
            fp2 = AllChem.GetMorganFingerprint(mol2.rdmol, 2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        if fingerprint_type == "rdkit":
            fp1 = Chem.RDKFingerprint(mol1.rdmol)
            fp2 = Chem.RDKFingerprint(mol2.rdmol)
        if fingerprint_type == "maccs":
            fp1 = MACCSkeys.GenMACCSKeys(mol1.rdmol)
            fp2 = MACCSkeys.GenMACCSKeys(mol2.rdmol)
        return DataStructs.FingerprintSimilarity(
            fp1, fp2,
            metric=DataStructs.TanimotoSimilarity
        )
    except Exception:
        return 0.0

def check_identical_molecules(mol1: Molecule, mol2: Molecule) -> bool:
    # Check if the two molecules are the same
    try:
        mol1._add_rdmol()
        mol2._add_rdmol()
        return Chem.MolToInchi(mol1.rdmol) == Chem.MolToInchi(mol2.rdmol)
    except Exception:
        return False

def mol_array_to_conformer(conf: np.ndarray) -> Chem.Conformer:
    new_conf = Chem.Conformer(conf.shape[0])
    for i in range(conf.shape[0]):
        new_conf.SetAtomPosition(i, tuple(conf[i]))
    return new_conf

#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
def calc_sa_score(molecule: Chem.RWMol) -> float:
    global _fscores
    if _fscores is None:
        fpscores_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "molecule", "fpscores.pkl.gz")
        data = cPickle.load(gzip.open(fpscores_file, "rb"))
        _fscores = {}
        for i in data:
            for j in range(1, len(i)):
                _fscores[i[j]] = float(i[0])

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(molecule, 2)  #<- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = molecule.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    ri = molecule.GetRingInfo()
    nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(molecule)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore

def calc_mol_diversity(mols: List[Molecule]) -> float:
    # Calculate the diversity of a list of molecules
    # Use the fingerprint similarity to calculate the diversity
    # The diversity is the average of the fingerprint similarity of all pairs of molecules
    dists = []
    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):
            mol1 = mols[i]
            mol2 = mols[j]
            mol1._add_rdmol()
            mol2._add_rdmol()
            dists.append(1 - molecule_fingerprint_similarity(mol1, mol2, fingerprint_type="rdkit"))
    return np.mean(dists)

def calc_mol_rmsd(mol1: Molecule, mol2: Molecule) -> float:
    # Calculate the RMSD of two molecules
    try:
        if mol1.conformer is None or mol2.conformer is None:
            raise ValueError("Conformer is not available for RMSD calculation")
        assert mol1.get_num_atoms() == mol2.get_num_atoms(), "The number of atoms of two molecules must be the same!"
        mol1._add_rdmol()
        mol2._add_rdmol()
        return Chem.rdMolAlign.CalcRMS(mol1.rdmol, mol2.rdmol, maxMatches=30000)
    except Exception:
        return 1e4

class MoleculeQEDTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return "Calculate the drug-likeness (QED score) of a molecule"

    def run(self, molecule: Molecule) -> float:
        return molecule.calc_qed()

class MoleculeSATool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return "Calculate the synthetic accessibility (SA score) of a molecule"

    def run(self, molecule: Molecule) -> float:
        return molecule.calc_sa()

class MoleculeLogPTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return "Calculate the solubility (LogP score) of a molecule"

    def run(self, molecule: Molecule) -> float:
        return molecule.calc_logp()

class MoleculeLipinskiTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return "Calculate the number of lipinski rules that a molecule satisfies"

    def run(self, molecule: Molecule) -> float:
        return molecule.calc_lipinski()

class MoleculeSimilarityTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    def print_usage(self) -> str:
        return "Calculate the Morgan fingerprint similarity of two molecules"

    def run(self, mol1: Molecule, mol2: Molecule) -> float:
        return molecule_fingerprint_similarity(mol1, mol2, fingerprint_type="morgan")
        