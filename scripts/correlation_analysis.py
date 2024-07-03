import sys
import scipy.stats
import pickle
from collections import defaultdict, OrderedDict
import pandas as pd
import statistics
import json
pd.set_option('display.max_columns', None)

def extract_allowed_session_ids_from_shared_prompts():
    json_file_path = 'data/shared_with_WG/wildbench_rubrics_w_add_info_id_only.json'
    with open(json_file_path, 'r') as file:
        info_dict = json.load(file)
    allowed_session_ids = info_dict["session_id"].values()
    allowed_session_ids_set = set(allowed_session_ids)
    assert len(allowed_session_ids_set) == len(allowed_session_ids), "Found duplicate session IDs, violating assumption that each example has a unique session ID."

    allowed_ids = info_dict["id"].values()
    allowed_ids_set = set(allowed_ids)
    assert len(allowed_ids_set) == len(allowed_ids), "Found duplicate IDs, violating assumption that each example has a unique ID."

    return allowed_ids_set, allowed_session_ids_set

# Banned words: set(["ethical", "inappropriate", "sexual", "I apologize", "harmful", "I don't feel comfortable", "I am afraid", "violence"])
allowed_session_ids_from_our_prompts = {'7cff0f95b819cf4cd9bbdbfb3ff463011', '7ec54a5ce036d201f668dfccb1803e351', '918a90aed332f87fc99a58a49433bc631', '3b04380721dce425479f60653e99acbb1', '9dc55ebf0bddda5b900d7c2cc4a605d41', 'b75cf148a68af2f68bfb3759cfd126da1', '553b027a6640db7d7d7f593c63dc9ded1', '1ecae15b328b64a27c8aa5ecc34dbaa11', '032bb4631908d47c1021432da601f2bd1', '064336a24025eaaefdaed036611869b71', '6dd638f49c38fb0206004e444abe59011', '547f4abc0f7d3ab02392c3aa35ebd6f41', '8a6e4d91fee78abee32f2aa7f1d5e5ea1', 'b78d9b096663be77e35e9e4bf00d3f001', 'ee663ffb8dabd92ca57c2fa67351ee6f1', 'c0a24bc107e16949f0d00a21f96bc5661', '4e5d470d0568b733175ea30e242849801', 'f89e112ed01429d2be2172a83c6e9d2c1', '34a603db02429424ffc50347f08fdbba1', '76384b75e7b04636902feea779a564c21', 'e6cd50977050ff1b936d3481f06c2dfc1', '3cc89681715e3c89b5754fb1615386081', '4ee0cc3818ae030fe0781c80451450b41', '3c55f0129b75da73dd9c907a6c4324a21', '3922bd7dd3c764ba52d28deab77fcc811', 'f48b09bc736d5067cdf8a9883f0033181', '61b2c75832a8debaa39472efe80cf3261', 'bff8f3dd1371ee03c7d9468e1cc5e3b21', '98f47d7d42a6d67d6941abcf9f11b3a41', '582df5e93ac79073de7fa6caa61cac1c1', 'a537a576542a02e1194be483e8fd8f021', '239180078c976d63f6a1646a103a9a8c1', 'e0d9e5dc05744fe1d7fd4cce257932301', '1a764da9d4d623efcc07cbcf7f8c3db51', 'cbe493ee2f9c175b01254315ff0cc26c1', '27a8c3528a77a78f3263e4e6305e833d1', '126faed4f80f62cf16c173afdffa1ba81', 'd52fe665c9d615ad1e3f1d1931786fce1', '328591c7cbbbf0fd65d16c7f73ccafc81', 'f22388683f18751017b4f71f0e896f731', '393c0cb462a7bb86c03590a6aded86281', 'b7ad287747ddcd834ddf27fbc0caa8511', '204cfd8d43543390eca1d461da17f4fc1', '8860af5fab441f23013b9f2a2ce3e8cd1', '37458363f6b78a6b967fed38e5cbca1a1', '1e37c9a56869a726175a4a41b579cee61', '48cc146ccf75d5e3557221836bb72b351', '2c82bf732545f9cfd7c09e0af59b4d711', '4d2b868fa0add8509293083a6ccbcd751', '77cac25262829509a5b2b4d4a5bd07291', '73a9b4493163433c1b1addb3734e49611', '0ac3fc8c5d3a9b04546eaf8edb8a52321', '4999a4452a4e565a900b3f2b9fb9ac5d1', '32272971a3ac8545d4322af9515de1f41', '60fc250407f5969ef17c7398e7695a151', 'e8d8e6f4e6b381d72b05e2efd113a7141', '8581d683421a3c4ea35965b6a5e55fdf1', '7bee143d4da968a94f08f57da6e465841', '98da0c7bababd5974a180504325714811', 'bd4cb0682194b6ecffa3ebeef72a59761', 'c6df30f11193f2cbb4ea2d66df713c041', 'f087bf014b4ed4f64acacd18e6b9f7d01', '3f2e8a0f0b777e128d7a0ba6287063451', 'bc6c8b2f2503598db5b7b5288d4496c91', 'ad6543a812c2d580434d1e822c8468551', '7fc6b3b14b26e327bb2f08461571afad1', 'fade06ca2da448816693241741c0026c1', '9b96dfca0890fc6e6d3daefadf3124bc1', '68a4fa5c3d2a5e2b9d308b4903514ad41', '6e03fd55d252723d64b201d8d4e8eeb71', '5b82c2826634ca112e4fdea092eccb681', '60d02f634af3ae4617e25843e032e81a1', '31ebe8ac434b39e358fe22eb3b3edbf51', '1ff7f34b204d0b9c72a610d221fdf06c1', '44ad058f52d1d48a5f7e919cef487ded1', 'd046164752080e81df503fa0859df1cb1', 'f8a150923d8f427c295d29fe3746b54c1', 'd048bf4d8b7bdf04ec03ff8c9e5db5591', 'a9ff6c9003fb6df2906f198ff6b23d5e1', '433d66cafe828b4ee7ac9069f75f10841', '34f80befa26b7e76aa46d8f840f66e4b1', 'a6d0a9faca4951da584467ed99b9c63b1', 'd910819a9d74c527352a6af461f2f8091', '995b5128fafd61f9f187fab2eabde3301', 'ca39a7b54eab2d265701fb8ea88c26e41', 'dd250b26b6d47fd443825e0cef8675931', '4e710835d76aa0c2962ec1d0a066909c1', '0e342dd56e8442655d3efacbd97a89871', '0fdee3ce4de526a5fff8e3e38d53bc971', '3d435c74377fc9701e4934888e06a9cf1', '4734fa7442008051cfa46057cf99aa421', '195ca389adcc67d3c16f9e18c5151d181', '06993db28ebd5349cd61bc596766fe851', 'd9d6ee3216dab4cc3f28f3589ccca73f1', '067c702443b8ffaebc51e277b6609a7c1', 'e2251d346a67bea6afcb3943a8d9ae571', '1b62c99b39eb329318413470dd73c0a31', '6f3bca4d27c73e718d507b31257a4c071', '26c78a5eaeaf6c15f03e268580f38ed11', 'c3c6f3a8d17ece51cce8ff65e650c5931', '7af665a1cffbf6387c2887d645e10d1b1', '7f7895aa68ea6362ecbd8d1e821801401', '6fa8b9bc3f54770183c810af3e384e3b1', '1e974936604a303a71959f55a348e08c1', '41103421274cd667ca1a2d66a73860771', 'b9e46b0be540a727b1bc973accf1a38b1', '002d6aa4153d566f1baa6331d279430a1', 'e07c897ce7e307b3b92e7e43021eb1e71', '1b83e4a32b0aa36452c9a4d68c196d451', 'b78b13012e9a27c07c99ed1a0e8ac22e1', '0f94707400a1555b1cc796975dcbbf911', 'b0b644cd482e72e10e310eb522645efd1', '6d47b64551da9f2dc3da3b0c11c9e32f1', '87bf5469bf25632ee70ee986661fcf9e1', '19d8929c9222729f0e0b1fcfe9ec08921', 'abed43c5d80de49789151f3fa42635871', '510707bd819d24814b45519382b1b9781', '40b203c3e39c983e22e67cbdccfcf2331', '37fdd68393f6a8bcb532474386caecc21', '6639019f56522ce980dcf287911c924d1', 'e0dc85c93930fbb6b7df71be59c911391', '41aef1286c50c4b01dd0ed26d5bf85841', '7654eb2d888da7189781aba40f9a15b51', 'afdff6576dc547b55d92ce77272e91e31', '44eda686dc89ddc4ebb0f281644bd7d91', '3685088c4281d442d29a5feea159aa3e1', 'f2730e25ea83fd405a3235659288e57b1', '4f5356f13c5102b6c01251b3187867c51', '6e001a3cb65bda276c0ebfcc37e4d5b21', '67b6bf290deecb1f13b3a2e6c4e15ebe1', 'ad62c4e25b1675544f063ead6b9c82f81', '32e9224cd2d5427ef3cf1fc4f248f6601', 'cdf94726ae5bf98737abfb9b45fb9aa71', 'cd62764127378e5226b3c4eec311af5c1', 'd9ca490a0611e5c6fa00aaedd8a60c671', 'eb3b227d1338a9cb6084db4d7fee46821', '82770ec7de23b5e1432d44cca2937bde1', '2b82fa20459648ae4fd9351266e23a531', '8e0fa4ac2a95d6c3ae96b17d9b031d5b1', '1fe45bd66e2e3e078b026ec43f51a9c61', 'a6fa2b0bf1a26dfa7ad2a2d7f63690501', '9c53dcafed65b6d73d0e3187850312911', 'd1614cd41f05fb23e1e13a3a6719af061', '51ae201868b35c3f40a49ea9268c13b51', '31a95239c54ad2d0d6aec2ebec4b91c81', '6e131d5fdb1ee87d0f505fc9e43f4bfc1', 'd3b338ea10a6a6871900e0056c24224b1', 'aae5bdfc6511f10e8a9c00e0887c084e1', 'a9a524a52eb9997e1a838a04f64b93691', 'c93293ead27309809b458b09b51298c31', '0891e7fc2d49b31b59ac70bc264998341', '991c2e873f18fc327c8e32bf6796945f1', 'c874d346d481b66de38b086dee90d74a1', '3b84d757b9d6077ae008cc582606efd71', '3c1c6e5799c789e7fc9cce07a35962991', '71c0e60021d2faf15a7c03b9a672e05c1', '1f8953b9851d165e90fd21393ae196951', '6daed63e86d823a0459242d28281787b1', '921259f697836fc81fd9e5986df805241', 'de24fb2c2566c29359f8e5987cf52d8a1', '4b7285146bd88fa060a85a83239f29ad1', '9cf738608478ba730fc5cd9f7f52a5941', 'cf5fd9b8ec7d2b336b59bfcdaab948781', '35c627621c8808ff6435328c4e0005441', '4a54422978e607e56da5cd4ed02e09b81', '6418109e5a0815d0cf0a9e4ddda204841', '1e83035d9885f3e2db24a0f63b0fc55c1', '659c45a3134555b5829282a6c65f459b1', 'abccd70fb13f7e5dd81aa8f0ab95ef451', 'cafb6a1555ebf27af6e802402e70acb91', 'a2e70da0127b39370d1c9b8f557251271', '620dd5ab873957275dd17912683907431', 'adf3dd1deae6fa13f6fe29e52cf5c9281', 'aa60c85eeae944f365de7019a866c3f41', '3b3194dbffe3bc47ac48889582502bd81', '9aaecea45d78f4182c54ab642834c48b1', '3c13a5c78606cf7e98712fecccaef88d1', '06d54358e79eb5246a87c01c095922de1', '5bb4e319cb7f38b4ee3f4497a410dd491', '6c66ae6a2bbefd1f06798c83eeedcf241', 'dd77122ee19e66cd8fea1502faefe45d1', '19218dc578d692076b37827823cdd5b71', '8f442621a3a60cc1c795b8e2760655551', 'ab8e4be9ec4b448a68a9949cd1ae746c1', 'a7bcd9e0a2873f067641b527c6f4abb71', 'c8cd31a7fb8ce59c66383a04c04410f31', '1f5b31e1dead1678e5c2c529802ba3791', '7c12ea7b1c7370e31edc32999df469cc1', 'c8d4d0debd88af25e45f9bf5f846bdc11', 'f0eb37b0b9037388c0558504ba58fa491', '6a5daefe98dca2e28f7895ea8afff0de1', '4ffddf9a46c801e6d924d6d5d4e2b8b81', '6a02834a607ed3416f32c5bee7977f5c1', '47dc25259c9917e91722cafc25c1fffd1', '8b6c3421286f011a2a38a87590c9298e1', '88596ac395c9c2dcff597fd8fd238dfa1', '71938ede5228e6be4fd4132bef11ef8c1', 'ed5f2606e7132a1e7f2e2acf51746a6b1', '393381827e2dd0180728ddecb8bda2c81', 'c21898698346eb8913b91391d973bbde1', '21bddcbda4cc5eeb6add8c93648d498e1', '610834ef8ff8d0454f6ad5cfa639922f1', '5a8f2ac0a4c4c491371d424d784318461', '6fe5ef9a2c964cd5176bd2da1a0c052f1', 'c86509d4954703978e22827638045c881', '5d3a05e107268d184769211cc6b1c7031', '02cecefc009040069f75eb244e5443fd1', '18932453967774c24af61e07796425b91', '953069514c9e440dcda463aa0197e1ca1', 'af8ea6d79a2dcc631f824b1fb671f1d71', '5ace00976fbda98e2ac7d4621aa028281', '157bdf2b552796ba441d352159fbc61d1', 'b5890a35eae61a7f1db08863e5a7f9871', '2a205fa062a1837ca1d31884b3c4c8881', '6171d3a500b2b5e7589d9baa96adf5e51', 'a9303ea72306968670b1313423e64d561', 'e22d84d6edeb65d757671afe0983ebfc1', 'ae8e1db3bb32cc4ad36354d80a2eb1f31', 'de0bb1744455ff78484c1511bfb660431', '67b97bf99619ef013d94e8ef08bb57561', '7bf94cb51c06158b3fa196cfb4f2ec8d1', '9c11a4a81cb744d6b2d4064470e65bd81', '4bea224f2283f88068ad0247c7cff8751', 'dd4ecb87773ae272953d8c3bd87de44d1', '8cea039fabd77b235a6e840e2b8645f81', '17d0015e5c392f927ffe83aa73abce501', '793dd6f680cb52c786d0412611d3027b1', '22e2e09e1d5578d0be7d187e2cfbf89c1', '9a213876f3dc4592705cae5c9f709a641', 'd4ed6ddecfd23bf2fdd366b45938a6e91', 'f9d2b0cc1c61924c213292ea6351a6d91', '26003567bd9d296f9f6f6b4c57d7015c1', '45f2465fb6661ce2bbb491a4c523e5a21', '2363124df66927aaecbf61d0adda5eed1', 'fc99c852d23936a26258794a2a8664241', '8528a258d7238b922f188da6569263401', '223814c0a065b6beecdaca6b268b4a381', '88f1eda16c48e74d2fb50bc8f150aa021', '98be6c98d2569872e08d73bc6a251b7d1', '156f41f3f6f5e27c1840b72e7ffc1dc71', 'cc023cb0c5767a6f48277dcf856b64241', 'f866c802dc703f81ddb5463a400b4c2e1', 'a87ecf8ef8ef22ebd4f8a100388a57d81', '2433aef343054f26cda551fc0a6d2efc1', '92c875f26ca916d004dce46742f8e5f41', '265edc10990a66a378aa5204bb99f68a1', '07eedbbc4b3b8d09cf59c4a37c87a50d1', 'b41a2ff59e35c3f692fba66b809086b81', '27008bf30ad4f7aa14e536fec52b34571', '3a769d191e79dee73b16a50d036e20be1', '6552f32e1159cc602cf11e8db6b982eb1', 'd817070c81c6dd32c985495d26879a611', 'e619320b8ec2f912a5bbd10279aa22fa1', 'd729a42f5a9d5f5adb96e80756f834a41', '50432b76e9376740be2318a417378a5f1', 'd4e2dee77bd0907883dcf7dafc9036a71', '211c0f0a2bce277b5a38f10f206698f21', 'c3589223bca207a2cd09b06892dc1efb1', 'b60ed301491227be8096fe0f4b85ea541', '0b5eb2e8e9050953015b2abdf8defea61', 'f4f6cc8228ae84e30bc344c925dc20821', '8e7878c9eee29520fea3f626fe4d77961', '639c3d6801dad89151da1cd5586612461', '3c0b8f2727fba2e7e46bca673eaf27c61', 'f872b2461d867a4a8b03563b3c89b0a71', 'ec792acd29daebad10e60ba18e50992e1', 'd14d6b23f18d2bfc4ab47c9d37d12d771', '1eca4bf2dabe52442542542a36efb9411', 'da14074b989371c7b0424c56697dae5f1', 'f461ac8f72ce59ac80b6cf50933301541', '8a175abe498a83cb2550ace7057940191', '12e7d55ae4be0b708cf2265f3ea2459f1', '74e31dbe8ea9aec3d4fecd5bf16997c51', 'a564f9589277ececac83f2c8f6fdb3451', 'a2e684c0e3bc49772f570d3b23eeb4051', '5b26a4df94828091b0afa6032d1fa2c11', '211e964a8b3cf6ace1eb218aaf5f24c01', '5092217507dd0f447a0bf5858542a4ef1', 'be5887c5d6c300c6a1dd9eb3b7f7e9e01', '638b223e81c6b73bd9d1cedda10ad76e1', '9fdf9d3c140e50f54d61f97a9e30c9c61', '1333d96a3e338a9f63eeba1be619165e1', '25931ecf274d266bbff32d8f286f86e31', '43f12309bc8576a34ce7fa8c1092b31e1', '736b4c6d4aad59ec652249d9e0e4cafa1', 'dd5949051e5482ad5f7378b3645838ce1', '2f61d2563bf3a159963d12a5d33c874f1', '59ad909089dfab40c01bc81149be1b041', '55ccefb161063a3924eeee5ae77a21701', 'a9765e3dfd9e7e4c1fc68c360697458a1', '532b869e98dbec76e09fcc233a3c06931', '264084adcd764be6a5d894ea43fbd6b71', '2429fc1b993e523b3677c03bbaa29e741', 'a94beea7aa69b8c0fe294353fa9e9db51', '51e2efe28c2b51671b541ea1a000f3331', 'cfa30d91f153e3a01ffe8ca6035194af1', '7aebe48f3422ce8715ce25af0dbac4f81', 'e51eb9b286e435ee0eeee35008ea45b21', '0396669a1b6fa8060e2d7dddf244e8031', 'ba84d9d18b1ed77541730a9d55c86e1b1', 'e56017e5032ba86e9d3ae82816854e621', '7d04ef952ed3de45758b37ea76d5d0b51', 'd2bdad24065dce139890f4dae5a1b5191', 'ab9efa6da5f6d4aa046ece220c78c9501', '83d3cada06976040be21a43f36aeda401', 'f7459b38da954ac20372355e066053bb1', 'bcdbaa5971a4996bf3f599af01729caa1', '0c50b8c9d273b0fca5eab53c76dffaa91', 'a2e13238e6a172678db9c7bca604d20c1', '6dae68451f914fe0864cc88b30c2c9501', 'c2dfc35b3fa144a9ba4aabc58ae9c19b1', '52431c3e328d5f0533b348f1941283ac1', 'fa1bc4eaa7b7a6a6e8b21979636da5071', 'de11fe4214cc35d3596e4582938c431c1', 'f6c6ec7381307339c7842874fd57cf071', 'b58d2a2751932f6af2fa2b41e41df88f1', 'cc248514d06562196fe60088acfa0f731', '0c5c06b2d797b4c2ed54dbd755f219091', '0e02595692e87b7e284179eddc3b3ef21', '3ef96223b393ca119eddf59ee910e8e21', '55e254aeb9a64cecfcf8d729fb2f0a2f1', 'bff5fb5d264d6fa6b015c03f5cd963ca1', 'c10a28be721ad78017083c0770bde2091', '18fa2e7c3a30c8029afd53e0504e8be91', 'afa43c61bcdf2f4bee9316a42cb7a4131', 'b9434777db76a059a631bbb1b3de94571', 'f448ac6a4fc66ec1ed5ad769649b99a41', '29bd9d45a868ecdaa47005bb3e7818051', '5f689f606b869d8619e3bff95d1c9a591', '68b9fff621d8486d0188711d690d0ddf1', '8b4842b762647b8a0b8d95957a4e4c711', '92701854bdb13c987c16174227c1a11a1', 'f73a37bca83f353d5834a21f5028884d1', '7ffe4236f58773b3288ffe6587fa11ef1', 'a083b0baa2b24eb29a7596f9830086271', '4249b5c7e19ca863faab3d3af9fdd7fa1', 'f57a81d53b91201dfe3fbb58653893731', 'c886b9dbbcaee3abb2e5a7937dbdc6021', 'a4659c791f2ab9255606dad56995de581', 'f41630dcb401ab7145c2a6f23d1a8b3d1', 'ce5182b0e827cd204c42ff90d2e3c3cf1', 'e654fb42d0fde99b616f5b1a10e9d3431', 'f92059b0c4d078acddd6efcc9a3a261a1', '69104d771e72c0cf69fabdcb4cbd11fe1', '56f3df94c0b205f0927e1d68bd0e1fcc1', '4d3962d5405b38476342f06c9d8717d61', 'bf9f40be909c1cf8dc0880155353d66d1', '731a31394c48025f101053c6ac46dfb31', '8bd0b3628678083cd70e97de9b1330301', 'f7371cbd2e16ef8c9e58eb5e596d483a1', '76fdfbc23691b2edf262bb250e1e69bb1', 'd4343c84f9caaf56bd94873deef42df61', '710a604c6698c4dcc3918db5b8fcf0b31', 'ab4c40187fb8ce55f71df8533f06c37f1', 'cfd1522038765ec829f4256db870e8d81', 'f85532e92c4c66a84fb921ce457ba4e21'}
_, allowed_session_ids_from_shared_prompts = extract_allowed_session_ids_from_shared_prompts()
# only_in_our_prompts = allowed_session_ids_from_our_prompts - allowed_session_ids_from_shared_prompts
# print(f"Only in our prompts: {len(only_in_our_prompts)}")
# only_in_shared_prompts = allowed_session_ids_from_shared_prompts - allowed_session_ids_from_our_prompts
# print(f"Only in shared prompts: {len(only_in_shared_prompts)}")
allowed_session_ids = allowed_session_ids_from_shared_prompts
assert len(allowed_session_ids) > 0

def load_evals_and_clean_data(category, completor_model_name, *, run_idx=0):
    evals_RubricEval_filename = f'outputs/global_rerun_exp_v5/{category}___{completor_model_name}___gpt-4o-2024-05-13_CoT_v0___evaluations_{run_idx}.pickle'
    with open(evals_RubricEval_filename, 'rb') as file:
        evals_RubricEval = pickle.load(file)
    evals_NoRubric_filename = f'outputs/global_no_rubrics_rerun_exp_v5/{category}___{completor_model_name}___gpt-4o-2024-05-13_CoT_v0___evaluations_{run_idx}.pickle'
    with open(evals_NoRubric_filename, 'rb') as file:
        evals_NoRubric = pickle.load(file)
    evals_HELMIns_filename = f'outputs/global_HELMInstruct_generic_rubrics_rerun_exp_v5/{category}___{completor_model_name}___gpt-4o-2024-05-13_CoT_v0___evaluations_{run_idx}.pickle'
    with open(evals_HELMIns_filename, 'rb') as file:
        evals_HELMIns = pickle.load(file)

    # evals_RubricEval = evals_RubricEval[~evals_RubricEval['session_id'].isin(banned_session_ids)]
    # evals_NoRubric = evals_NoRubric[~evals_NoRubric['session_id'].isin(banned_session_ids)]
    # evals_HELMIns = evals_HELMIns[~evals_HELMIns['session_id'].isin(banned_session_ids)]
    evals_RubricEval = evals_RubricEval[evals_RubricEval['session_id'].isin(allowed_session_ids)]
    evals_NoRubric = evals_NoRubric[evals_NoRubric['session_id'].isin(allowed_session_ids)]
    evals_HELMIns = evals_HELMIns[evals_HELMIns['session_id'].isin(allowed_session_ids)]

    common_final_prompts = defaultdict(int)

    for prompt in evals_RubricEval['final_prompt']:
        common_final_prompts[prompt] += 1
    for prompt in evals_NoRubric['final_prompt']:
        common_final_prompts[prompt] += 1
    for prompt in evals_HELMIns['final_prompt']:
        common_final_prompts[prompt] += 1

    # Only keep the common prompts
    common_final_prompts = {key: value for key, value in common_final_prompts.items() if value == 3}

    evals_RubricEval = evals_RubricEval[evals_RubricEval['final_prompt'].isin(common_final_prompts.keys())].sort_values(by='final_prompt')
    evals_NoRubric = evals_NoRubric[evals_NoRubric['final_prompt'].isin(common_final_prompts.keys())].sort_values(by='final_prompt')
    evals_HELMIns = evals_HELMIns[evals_HELMIns['final_prompt'].isin(common_final_prompts.keys())].sort_values(by='final_prompt')

    return evals_RubricEval, evals_NoRubric, evals_HELMIns

# category = sys.argv[1]
# completor_model_name = sys.argv[2]
model_names = [
    "gpt-4o-2024-05-13",
    "chatgpt",

    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",

    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    
    "Mixtral-8x22B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.3",
    # "Qwen2-72B-Instruct",
]
categories = ["Advice seeking", "Reasoning", "Editing", "Planning", "Information seeking", "Creative Writing", "Coding & Debugging", "Brainstorming", "Math", "Role playing", "Data Analysis", "Others"]
# categories = ["Math"]  # rank cor: 0.94; excluding Claude: 0.92
# categories = ["Coding & Debugging"]  # rank cor: 0.9; excluding Claude: 0.87
# categories = ["Reasoning"]  # rank cor: 0.8; excluding Claude: 0.92
# categories = ["Brainstorming"]  # rank cor: 0.8; excluding Claude: 0.87
# categories = ["Editing"]  # rank cor: 0.78; excluding Claude: 0.97
# categories = ["Data Analysis"]  # rank cor: 0.78; excluding Claude: 0.88
# categories = ["Planning"]  # rank cor: 0.76; excluding Claude: 0.83
# categories = ["Information seeking"]  # rank cor: 0.61; excluding Claude: 0.92
# categories = ["Creative Writing"]  # rank cor: 0.58; excluding Claude: 0.85
# categories = ["Advice seeking"]  # rank cor: 0.5; excluding Claude: 0.52
# categories = ["Role playing"]  # rank cor: 0.39; excluding Claude: 0.75
# categories = ["Others"]  # rank cor: 0.15; excluding Claude: -0.12
# categories_meta_list = [[x] for x in categories]
categories_meta_list = [categories]

for categories in categories_meta_list:
    print(f"Categories: {categories}")
    RubricEval_vs_NoRubric_category_to_cors = defaultdict(list)
    RubricEval_vs_HELMIns_category_to_cors = defaultdict(list)
    NoRubric_vs_HELMIns_category_to_cors = defaultdict(list)

    RubricEval_avg_score_vector_global = []
    NoRubric_avg_score_vector_global = []
    HELMIns_avg_score_vector_global = []

    RubricEval_model_to_avg_scores = OrderedDict()
    NoRubric_model_to_avg_scores = OrderedDict()
    HELMIns_model_to_avg_scores = OrderedDict()
    for model_name in model_names:
        RubricEval_model_to_avg_scores[model_name] = []  # List of avg scores for each category
        NoRubric_model_to_avg_scores[model_name] = []
        HELMIns_model_to_avg_scores[model_name] = []

    # print(r"""
    # \begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
    # \hline
    # Score & GPT4o & GPT3.5T & Gem1.5P & Gem1.5F & Claude3-O & Claude3-S & Claude3-H & Llama3-70B & Llama3-8B & Mixtral-8x22B & Mistral-7B & Qwen2-72B & Avg. \\
    # \hline
    # """)
    for category_idx, category in enumerate(categories):
    #     print(r"""          
    # \hline
    # \multicolumn{14}{|c|}{Category: $CATEGORY} &
    # \hline
    # """.replace("$CATEGORY", category.replace('&', r'\&')))
        # print(f"Category: {category}")
        for model_name in model_names:
            # print(f"Model name: {model_name}")
            evals_RubricEval, evals_NoRubric, evals_HELMIns = load_evals_and_clean_data(category, model_name, run_idx=0)
            RubricEval_avg_score_vector_global.extend(evals_RubricEval.avg_score.tolist())
            NoRubric_avg_score_vector_global.extend(evals_NoRubric.avg_score.tolist())
            HELMIns_avg_score_vector_global.extend(evals_HELMIns.avg_score.tolist())
            RubricEval_model_to_avg_scores[model_name].extend(evals_RubricEval.avg_score.tolist())
            NoRubric_model_to_avg_scores[model_name].extend(evals_NoRubric.avg_score.tolist())
            HELMIns_model_to_avg_scores[model_name].extend(evals_HELMIns.avg_score.tolist())

            # if len(evals_RubricEval.avg_score) >= 2 and len(evals_NoRubric.avg_score) >= 2 and len(evals_HELMIns.avg_score) >= 2:
            #     RubricEval_vs_NoRubric_cor = scipy.stats.pearsonr(evals_RubricEval.avg_score, evals_NoRubric.avg_score)
            #     # print(f"evals_RubricEval vs. evals_NoRubric: {RubricEval_vs_NoRubric_cor}")
            #     RubricEval_vs_NoRubric_category_to_cors[category].append(RubricEval_vs_NoRubric_cor.statistic)

            #     RubricEval_vs_HELMIns_cor = scipy.stats.pearsonr(evals_RubricEval.avg_score, evals_HELMIns.avg_score)
            #     # print(f"evals_RubricEval vs. evals_HELMIns: {RubricEval_vs_HELMIns_cor}")
            #     RubricEval_vs_HELMIns_category_to_cors[category].append(RubricEval_vs_HELMIns_cor.statistic)

            #     NoRubric_vs_HELMIns_cor = scipy.stats.pearsonr(evals_NoRubric.avg_score, evals_HELMIns.avg_score)
            #     # print(f"evals_NoRubric vs. evals_HELMIns: {NoRubric_vs_HELMIns_cor}")
            #     NoRubric_vs_HELMIns_category_to_cors[category].append(NoRubric_vs_HELMIns_cor.statistic)

        # mean_RubricEval_vs_NoRubric_cors = statistics.mean(RubricEval_vs_NoRubric_category_to_cors[category])
        # mean_RubricEval_vs_HELMIns_cors = statistics.mean(RubricEval_vs_HELMIns_category_to_cors[category])
        # mean_NoRubric_vs_HELMIns_cors = statistics.mean(NoRubric_vs_HELMIns_category_to_cors[category])
        # print(f"RE vs. NR & {' & '.join([f'{cor:.2f}' for cor in RubricEval_vs_NoRubric_category_to_cors[category]])} & {f'{mean_RubricEval_vs_NoRubric_cors:.2f}'} \\\\")
        # print(f"RE vs. HI & {' & '.join([f'{cor:.2f}' for cor in RubricEval_vs_HELMIns_category_to_cors[category]])} & {f'{mean_RubricEval_vs_HELMIns_cors:.2f}'} \\\\")
        # print(f"NR vs. HI & {' & '.join([f'{cor:.2f}' for cor in NoRubric_vs_HELMIns_category_to_cors[category]])} & {f'{mean_NoRubric_vs_HELMIns_cors:.2f}'} \\\\")
        # print(f"mean(RubricEval_vs_NoRubric_cors): {mean_RubricEval_vs_NoRubric_cors}")
        # print(f"mean(RubricEval_vs_HELMIns_cors): {mean_RubricEval_vs_HELMIns_cors}")
        # print(f"mean(NoRubric_vs_HELMIns_cors): {mean_NoRubric_vs_HELMIns_cors}")

    # print(r"""
    # \hline
    # \end{tabular}
    # """)

    print(f"Global: RubricEval vs. NoRubric per-model-per-category concat cors: {scipy.stats.pearsonr(RubricEval_avg_score_vector_global, NoRubric_avg_score_vector_global)}")
    print(f"Global: RubricEval vs. HELMIns per-model-per-category concat cors: {scipy.stats.pearsonr(RubricEval_avg_score_vector_global, HELMIns_avg_score_vector_global)}")
    print(f"Global: NoRubric vs. HELMIns per-model-per-category concat cors: {scipy.stats.pearsonr(NoRubric_avg_score_vector_global, HELMIns_avg_score_vector_global)}")

    # for model_name in model_names:
    #     print(f"Model name: {model_name}")
    #     print(f"Global: RubricEval vs. NoRubric per-model cors: {scipy.stats.pearsonr(RubricEval_model_to_avg_scores[model_name], NoRubric_model_to_avg_scores[model_name])}")
    #     print(f"Global: RubricEval vs. HELMIns per-model cors: {scipy.stats.pearsonr(RubricEval_model_to_avg_scores[model_name], HELMIns_model_to_avg_scores[model_name])}")
    #     print(f"Global: NoRubric vs. HELMIns per-model cors: {scipy.stats.pearsonr(NoRubric_model_to_avg_scores[model_name], HELMIns_model_to_avg_scores[model_name])}")

    RubricEval_model_to_avg_of_avg_score = OrderedDict()
    for model_name, scores in RubricEval_model_to_avg_scores.items():
        RubricEval_model_to_avg_of_avg_score[model_name] = statistics.mean(scores)
    RubricEval_ranked_model_names = sorted(model_names, key=lambda model_name: RubricEval_model_to_avg_of_avg_score[model_name], reverse=True)
    print(f"RubricEval_ranked_model_names: {RubricEval_ranked_model_names}")
    RubricEval_model_names_rank_vector = []
    for model_name in model_names:
        RubricEval_model_names_rank_vector.append(RubricEval_ranked_model_names.index(model_name) + 1)

    NoRubric_model_to_avg_of_avg_scores = OrderedDict()
    for model_name, scores in NoRubric_model_to_avg_scores.items():
        NoRubric_model_to_avg_of_avg_scores[model_name] = statistics.mean(scores)
    NoRubric_ranked_model_names = sorted(model_names, key=lambda model_name: NoRubric_model_to_avg_of_avg_scores[model_name], reverse=True)
    print(f"NoRubric_ranked_model_names: {NoRubric_ranked_model_names}")
    NoRubric_model_names_rank_vector = []
    for model_name in model_names:
        NoRubric_model_names_rank_vector.append(NoRubric_ranked_model_names.index(model_name) + 1)

    HELMIns_model_to_avg_of_avg_scores = OrderedDict()
    for model_name, scores in HELMIns_model_to_avg_scores.items():
        HELMIns_model_to_avg_of_avg_scores[model_name] = statistics.mean(scores)
    HELMIns_ranked_model_names = sorted(model_names, key=lambda model_name: HELMIns_model_to_avg_of_avg_scores[model_name], reverse=True)
    print(f"HELMIns_ranked_model_names: {HELMIns_ranked_model_names}")
    HELMIns_model_names_rank_vector = []
    for model_name in model_names:
        HELMIns_model_names_rank_vector.append(HELMIns_ranked_model_names.index(model_name) + 1)

    # ChatbotArena_ranked_model_names = [
    #     "gpt-4o-2024-05-13",  # Rank-1
    #     "gemini-1.5-pro-001",  # Rank-2
    #     "claude-3-opus-20240229",  # Rank-6
    #     "gemini-1.5-flash-001",  # Rank-10
    #     "Meta-Llama-3-70B-Instruct",  # Rank-11
    #     "claude-3-sonnet-20240229",  # Rank-12
    #     "Qwen2-72B-Instruct",  # Rank-15
    #     "claude-3-haiku-20240307",  # Rank-18
    #     "Meta-Llama-3-8B-Instruct",  # Rank-23
    #     "Mixtral-8x22B-Instruct-v0.1",  # Rank-27
    #     "chatgpt", # Rank-40
    #     "Mistral-7B-Instruct-v0.3",  # Rank-60
    # ]
    # ChatbotArena_model_names_rank_vector = []
    # for model_name in model_names:
    #     ChatbotArena_model_names_rank_vector.append(ChatbotArena_ranked_model_names.index(model_name) + 1)
    # Global
    ChatbotArena_model_to_scores = {
        "gpt-4o-2024-05-13": 1287,
        "gemini-1.5-pro-001": 1266,
        "claude-3-opus-20240229": 1249,
        "gemini-1.5-flash-001": 1232,
        "Meta-Llama-3-70B-Instruct": 1208,
        "claude-3-sonnet-20240229": 1202,
        "Qwen2-72B-Instruct": 1187,
        "claude-3-haiku-20240307": 1178,
        "Meta-Llama-3-8B-Instruct": 1153,
        "Mixtral-8x22B-Instruct-v0.1": 1146,
        "chatgpt": 1106,
        "Mistral-7B-Instruct-v0.3": 1068,
    }
    # Coding
    # assert categories == ["Coding & Debugging"]  # Coding only experiment
    # ChatbotArena_model_to_scores = {
    #     "gpt-4o-2024-05-13": 1299,
    #     "gemini-1.5-pro-001": 1272,
    #     "claude-3-opus-20240229": 1253,
    #     "gemini-1.5-flash-001": 1238,
    #     "Meta-Llama-3-70B-Instruct": 1202,
    #     "claude-3-sonnet-20240229": 1216,
    #     "Qwen2-72B-Instruct": 1186,
    #     "claude-3-haiku-20240307": 1191,
    #     "Meta-Llama-3-8B-Instruct": 1150,
    #     "Mixtral-8x22B-Instruct-v0.1": 1153,
    #     "chatgpt": 1137,
    #     "Mistral-7B-Instruct-v0.3": 1076,
    # }
    ChatbotArena_model_names_score_vector = []
    for model_name in model_names:
        ChatbotArena_model_names_score_vector.append(ChatbotArena_model_to_scores[model_name])
    ChatbotArena_ranked_model_names = sorted(model_names, key=lambda model_name: ChatbotArena_model_to_scores[model_name], reverse=True)
    print(f"ChatbotArena_ranked_model_names: {ChatbotArena_ranked_model_names}")
    ChatbotArena_model_names_rank_vector = []
    for model_name in model_names:
        ChatbotArena_model_names_rank_vector.append(ChatbotArena_ranked_model_names.index(model_name) + 1)

    # WildBenchv2_ranked_model_names = [
    #     "gpt-4o-2024-05-13",  # Rank-1
    #     "gemini-1.5-pro-001",  # Rank-3
    #     "claude-3-opus-20240229",  # Rank-5
    #     "Meta-Llama-3-70B-Instruct",  # Rank-6
    #     "gemini-1.5-flash-001",  # Rank-7

    #     "claude-3-sonnet-20240229",  # Rank-13
    #     "Qwen2-72B-Instruct",  # Rank-14
    #     "claude-3-haiku-20240307",  # Rank-18

    #     "Meta-Llama-3-8B-Instruct",  # Rank-24
    #     "Mixtral-8x22B-Instruct-v0.1",  # Rank-27 (this is a guess, not in the leaderboard)
    #     "Mistral-7B-Instruct-v0.3",  # Rank-30

    #     "chatgpt", # Rank-34
    # ]
    # WildBenchV2_model_names_rank_vector = []
    # for model_name in model_names:
    #     WildBenchV2_model_names_rank_vector.append(WildBenchv2_ranked_model_names.index(model_name) + 1)
    # # WB-Score
    # WildBenchv2_model_to_scores = {
    #     "gpt-4o-2024-05-13": 64.9,
    #     "gemini-1.5-pro-001": 55.8,
    #     "claude-3-opus-20240229": 62.1,
    #     "Meta-Llama-3-70B-Instruct": 59.3,
    #     "gemini-1.5-flash-001": 53.5,

    #     "claude-3-sonnet-20240229": 55.3,
    #     "Qwen2-72B-Instruct": 55.8,
    #     "claude-3-haiku-20240307": 49.8,

    #     "Meta-Llama-3-8B-Instruct": 43.7,
    #     "Mixtral-8x22B-Instruct-v0.1": 42, # (this is a guess, not in the leaderboard)
    #     "Mistral-7B-Instruct-v0.3": 40.1,

    #     "chatgpt": 40.7,
    # }
    # WildBenchV2_model_names_score_vector = []
    # for model_name in model_names:
    #     WildBenchV2_model_names_score_vector.append(WildBenchv2_model_to_scores[model_name])

    print(f"len(RubricEval_model_to_avg_scores['chatgpt']): {len(RubricEval_model_to_avg_scores['chatgpt'])}")

    print(f"Global: RubricEval vs. ChatbotArena rank cor: {scipy.stats.spearmanr(RubricEval_model_names_rank_vector, ChatbotArena_model_names_rank_vector)}")
    print(f"Global: NoRubric vs. ChatbotArena rank cor: {scipy.stats.spearmanr(NoRubric_model_names_rank_vector, ChatbotArena_model_names_rank_vector)}")
    print(f"Global: HELMIns vs. ChatbotArena rank cor: {scipy.stats.spearmanr(HELMIns_model_names_rank_vector, ChatbotArena_model_names_rank_vector)}")
    # print(f"Global: WildBenchV2 vs. ChatbotArena rank cor: {scipy.stats.spearmanr(WildBenchV2_model_names_rank_vector, ChatbotArena_model_names_rank_vector)}")
    # print(f"Global: RubricEval vs. WildBenchV2 rank cor: {scipy.stats.spearmanr(RubricEval_model_names_rank_vector, WildBenchV2_model_names_rank_vector)}")
    print()
    print(f"Global: RubricEval vs. ChatbotArena pearson cor: {scipy.stats.pearsonr(list(RubricEval_model_to_avg_of_avg_score.values()), ChatbotArena_model_names_score_vector)}")
    print(f"Global: NoRubric vs. ChatbotArena pearson cor: {scipy.stats.pearsonr(list(NoRubric_model_to_avg_of_avg_scores.values()), ChatbotArena_model_names_score_vector)}")
    print(f"Global: HELMIns vs. ChatbotArena pearson cor: {scipy.stats.pearsonr(list(HELMIns_model_to_avg_of_avg_scores.values()), ChatbotArena_model_names_score_vector)}")
    # print(f"Global: WildBenchV2 vs. ChatbotArena pearson cor: {scipy.stats.pearsonr(WildBenchV2_model_names_score_vector, ChatbotArena_model_names_score_vector)}")
    # print(f"Global: RubricEval vs. WildBenchV2 pearson cor: {scipy.stats.pearsonr(list(RubricEval_model_to_avg_of_avg_score.values()), WildBenchV2_model_names_score_vector)}")