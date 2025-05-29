def name():
    # FirstName = 'Ranjan'
    # LastName = 'Mistry'

    class PersonName:
        FirstName = 'Adrian'
        LastName = 'Brody'

        @classmethod
        def fullname(cls):
            # FirstName = 'CV'
            # LastName = 'Raman'

            fullname = f"{cls.FirstName} {cls.LastName}"
            return fullname

    return PersonName


cls = name()
print(cls.fullname())