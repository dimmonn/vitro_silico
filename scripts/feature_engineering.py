from rdkit import Chem
from rdkit.Chem import Descriptors


class FeatureExtractor:

    def __extract_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                descriptors = [
                    Descriptors.MolWt(mol),
                    Descriptors.NumHAcceptors(mol),
                    # Add more descriptors as needed
                ]
                return descriptors
            else:
                return None
        except Exception as e:
            print(f"Error processing SMILES: {smiles}. Error: {str(e)}")
            return None

    def get_feature_extraction_lambda(self):
        return lambda x: self.__extract_features(x)
