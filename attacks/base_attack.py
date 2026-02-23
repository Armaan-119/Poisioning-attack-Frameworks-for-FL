class BaseAttack:
    def apply(self, model, trainloader=None):
        """
        Modify model or training behavior.
        """
        raise NotImplementedError
