# Copyright 2020 FiveAI Ltd.
# All rights reserved.
#
# This file is part of the PaRoT toolbox, and is released under the
# "MIT License Agreement". Please see the LICENSE file that should
# have been included as part of this package.


class DomainNotSupported(BaseException):
    pass


class Property():
    SUPPORTED_DOMAINS = []

    def __init__(self):
        pass

    def of(self, domain, input_tensor):
        """
        Create an object of type domain which is trainable on the property for
        the given input_tensor.

        Args:
            domain (type): desired type of the output domain
            input_tensor (tf.tensor): input to the desired domain
        """
        if domain not in self.SUPPORTED_DOMAINS:
            raise DomainNotSupported(
                '{} is not supported by the property {}'.format(
                    domain.__name__, self.__class__.__name__))

        # create the property
        return self.generate_property(domain, input_tensor)

    def generate_property(self, domain, input_tensor):
        """
        Obtain the abstraction domain corresponding to the input tensor and the
        property.
        """
        raise NotImplementedError
