from rdkit import Chem
from rdkit.Chem import Descriptors

from scripts.base_logger import VitroLogger as logger

logger = logger()


class FeatureExtractor:

    def __extract_features_list(self, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                descriptors = [
                    Descriptors.MolWt(mol),
                    Descriptors.NumHAcceptors(mol)
                ]
                return descriptors
            else:
                return None
        except Exception as e:
            logger.error(f"Error processing SMILES: {smiles}. Error: {str(e)}")
            return None

    def __extract_features_mol(self, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                return [Descriptors.MolWt(mol)]
            else:
                return None
        except Exception as e:
            logger.error(f"Error processing SMILES: {smiles}. Error: {str(e)}")
            return None

    def get_feature_descr_list_lambda(self):
        return lambda x: self.__extract_features_list(x)

    def get_feature_descr_mol_lambda(self):
        return lambda x: self.__extract_features_mol(x)
