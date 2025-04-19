import pkg_resources

pyOpenSSL = "16.2.0"
requirements = pkg_resources.require("botocore==" + botocore_version)
for req in requirements[0].requires():
    print(req)