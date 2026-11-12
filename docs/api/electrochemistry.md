# Electrochemistry

```{eval-rst}
.. automodule:: sumfem.electrochemistry
```

## Materials

```{eval-rst}
.. automodule:: sumfem.electrochemistry.materials

.. currentmodule:: sumfem.electrochemistry
```

### Electrolytes

```{eval-rst}
.. autoclass:: Electrolyte
   :members:

.. autoclass:: Ion
   :members:
```

### Redox couples

```{eval-rst}
.. autoclass:: RedoxCouple
   :members:
```

### Porous electrodes

```{eval-rst}
.. autoclass:: PorousElectrode
   :members:

.. autoclass:: Region
   :members:
```

## Models

```{eval-rst}
.. automodule:: sumfem.electrochemistry.models

.. currentmodule:: sumfem.electrochemistry
```

### Charge and transport

```{eval-rst}
.. autoclass:: ElectrolyteConduction
   :members:

.. autoclass:: NernstPlanck
   :members:

.. autoclass:: TransientNernstPlanck
   :members:
```

### Kinetics

```{eval-rst}
.. autoclass:: ElectrodeReaction
   :members:

.. autoclass:: ExchangeCurrentButlerVolmer
   :members:

.. autoclass:: ButlerVolmer
   :members:

.. autoclass:: MassTransferButlerVolmer
   :members:

.. autoclass:: PistonVelocityButlerVolmer
   :members:
```

### Membranes

```{eval-rst}
.. autoclass:: ProtonExchangeMembrane
   :members:

.. autoclass:: SchloeglFlow
   :members:

.. autoclass:: MembraneProtonFlux
   :members:
```

### Flow through the electrodes

```{eval-rst}
.. autoclass:: ElectrodeFlow
   :members:

.. autoclass:: DarcyFlow
   :members:

.. autoclass:: FlowSetup
   :members:
```

## Cells

```{eval-rst}
.. automodule:: sumfem.electrochemistry.cells

.. currentmodule:: sumfem.electrochemistry
```

### Half-cells

```{eval-rst}
.. autoclass:: RedoxHalfCell
   :members:
```

### Full cells

```{eval-rst}
.. autoclass:: FullCell
   :members:

.. autoclass:: HalfCellSpec
   :members:

.. autoclass:: MembraneSpec
   :members:

.. autoclass:: FlowSpec
   :members:
   :exclude-members: model
```
