"""Unit test for SMP v2 image uris."""

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import pytest
from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris


_SMP_ENABLED_TEMPLATE = "658645717510.dkr.ecr.{region}.amazonaws.com/smdistributed-modelparallel:2.0.1-gpu-{py_version}-{cuda_version}"
_SMP_DISABLED_TEMPLATE = "{account}.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.1-gpu-{py_version}"


@pytest.mark.parametrize("load_config", ["pytorch-smp.json"], indirect=True)
@pytest.mark.parametrize(
    "instance_type_and_cuda_version", (
        ("ml.p3dn.24xlarge", "cu118"),
        ("ml.p4d.24xlarge", "cu118"),
        ("ml.p5.48xlarge", "cu121"),
    )
)
@pytest.mark.parametrize(
    "distribution_and_image_uri", (
        # Enabled with sample image uri:
        # - 658645717510.dkr.ecr.us-west-2.amazonaws.com/smdistributed-modelparallel:2.0.1-gpu-py310-cu121
        (
            {
                "torch_distributed": {"enabled": True},
                "smdistributed": {"modelparallel": {"enabled": True}},
            },
            _SMP_ENABLED_TEMPLATE,
            True,
        ),
        (
            {
                "torch_distributed": {},
                "smdistributed": {"modelparallel": {}},
            },
            _SMP_ENABLED_TEMPLATE,
            True,
        ),
        # Disabled with sample image uri:
        # - 763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310
        (
            # Distribution is None.
            None,
            _SMP_DISABLED_TEMPLATE,
            False,
        ),
        (
            # Missing `torch_distributed`.
            {
                "smdistributed": {"modelparallel": {"enabled": True}},
            },
            _SMP_DISABLED_TEMPLATE,
            False,
        ),
        (
            # Missing `modelparallel`.
            {
                "torch_distributed": {},
                "smdistributed": {},
            },
            _SMP_DISABLED_TEMPLATE,
            False,
        ),
        (
            # Missing `smdistributed`.
            {
                "torch_distributed": {},
            },
            _SMP_DISABLED_TEMPLATE,
            False,
        ),
    )
)
def test_smp_v2(load_config, instance_type_and_cuda_version, distribution_and_image_uri):
    """Unit test for SMP v2 containers."""
    instance_type, cuda_version = instance_type_and_cuda_version
    distribution, expected_image_uri_template, do_extra_check = distribution_and_image_uri

    versions = load_config["training"]["versions"]
    processors = load_config["training"]["processors"]

    for processor in processors:
        for version in versions:
            accounts = load_config["training"]["versions"][version]["registries"]
            py_versions = load_config["training"]["versions"][version]["py_versions"]
            for py_version in py_versions:
                for region in sorted(accounts):
                    uri = image_uris.get_training_image_uri(
                        region,
                        framework="pytorch",
                        framework_version=version,
                        py_version=py_version,
                        distribution=distribution,
                        instance_type=instance_type,
                    )

                    if do_extra_check:
                        # Extra util function.
                        expected = expected_uris.framework_uri(
                            repo="smdistributed-modelparallel",
                            fw_version=version,
                            py_version=f"{py_version}-{cuda_version}",
                            processor=processor,
                            region=region,
                            account=accounts[region],
                        )
                        assert uri == expected

                    expected_image_uri = expected_image_uri_template.format(
                        account={"ap-northeast-3": "364406365360"}.get(region, "763104351884"),
                        cuda_version=cuda_version,
                        py_version=py_version,
                        region=region,
                    )
                    print(
                        f"(porcessor, version, region, account, py_version, instance_type, uri) = "
                        f"({processor}, {version}, {region}, {accounts[region]}, {py_version}, "
                        f"{instance_type:15s}, {uri:50s} vs {expected_image_uri} ==> "
                        f"{uri == expected_image_uri})."
                    )
                    assert uri == expected_image_uri
