from tortoise import Model, fields


class DataCollection(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(50)
    collection_type = fields.CharField(50)

    def __str__(self):
        return f"DataCollection {self.id}: {self.name}"
