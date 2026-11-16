// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Reference selection and units for the shared molecular IGMH analysis.

use super::method::{run_collinear_property_reference, CollinearPropertyReference, MolecularSetup};
use super::molecular::{reference_orbital_residual, reference_report, setup};
use super::{Controls, System, Value};
use crate::properties::igmh::{analyze, validate_partition, GradientConvention};
use crate::{DftError, Matrix, Result, Vec3};

pub(super) struct Request {
    points: Vec<Vec3>,
    shape: Option<[usize; 3]>,
    fragments: Option<Vec<usize>>,
    convention: GradientConvention,
    atom_details: bool,
}

fn cartesian_rows(rows: Vec<Vec<f64>>, key: &str) -> Result<Vec<Vec3>> {
    rows.into_iter()
        .map(|p| match p.as_slice() {
            [x, y, z] => Ok(Vec3::new(*x, *y, *z)),
            _ => Err(DftError::InvalidInput(format!(
                "{key} must contain Cartesian triples"
            ))),
        })
        .collect()
}

fn sampling(controls: &Controls) -> Result<(Vec<Vec3>, Option<[usize; 3]>)> {
    match (
        controls.opt_rows("igmh_points")?,
        controls.opt_rows("igmh_grid")?,
        controls.opt_usize_list("igmh_grid_shape")?,
    ) {
        (Some(rows), None, None) if !rows.is_empty() => {
            Ok((cartesian_rows(rows, "igmh_points")?, None))
        }
        (None, Some(grid), Some(shape)) => {
            let [origin, x, y, z]: [Vec3; 4] =
                cartesian_rows(grid, "igmh_grid")?.try_into().map_err(|_| {
                    DftError::InvalidInput(
                        "igmh_grid requires an origin and three step vectors".into(),
                    )
                })?;
            let [nx, ny, nz]: [usize; 3] = shape.try_into().map_err(|_| {
                DftError::InvalidInput("igmh_grid_shape requires three positive integers".into())
            })?;
            let count = nx
                .checked_mul(ny)
                .and_then(|n| n.checked_mul(nz))
                .filter(|&n| n > 0)
                .ok_or_else(|| DftError::InvalidInput("invalid IGMH grid size".into()))?;
            let determinant = x.dot(y.cross(z));
            if !determinant.is_finite() || determinant == 0.0 {
                return Err(DftError::InvalidInput(
                    "IGMH grid step vectors must span three dimensions".into(),
                ));
            }
            let mut points = Vec::new();
            points
                .try_reserve_exact(count)
                .map_err(|_| DftError::InvalidInput("IGMH grid does not fit in memory".into()))?;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        points.push(origin + x * i as f64 + y * j as f64 + z * k as f64);
                    }
                }
            }
            Ok((points, Some([nx, ny, nz])))
        }
        _ => Err(DftError::InvalidInput(
            "IGMH requires nonempty igmh_points or both igmh_grid and igmh_grid_shape".into(),
        )),
    }
}

impl Request {
    pub(super) fn parse(controls: &Controls) -> Result<Self> {
        if super::double_hybrid::requested(controls)? {
            return Err(DftError::Unsupported {
                feature: "IGMH of a double-hybrid reference".into(),
                missing: "selection of a relaxed PT2 one-particle density for IGMH".into(),
            });
        }
        let (points, shape) = sampling(controls)?;
        let convention = match controls.str("igmh_gradient", "published")?.as_str() {
            "published" => GradientConvention::Published,
            "hirshfeld" => GradientConvention::Hirshfeld,
            _ => {
                return Err(DftError::InvalidInput(
                    "igmh_gradient must be published or hirshfeld".into(),
                ))
            }
        };
        Ok(Self {
            points,
            shape,
            fragments: controls.opt_usize_list("fragments")?,
            convention,
            atom_details: controls.bool("igmh_atom_details", false)?,
        })
    }

    pub(super) fn validate(&self, setup: &MolecularSetup) -> Result<()> {
        validate_partition(&setup.molecule, &setup.basis, self.fragments.as_deref())?;
        Ok(())
    }

    pub(super) fn append(
        &self,
        report: &mut Value,
        setup: &MolecularSetup,
        p: &Matrix,
    ) -> Result<()> {
        let output = analyze(
            &setup.molecule,
            &setup.basis,
            p,
            &self.points,
            self.fragments.as_deref(),
            self.convention,
            self.atom_details,
        )?;
        report.set(
            "igmh_gradient",
            match self.convention {
                GradientConvention::Published => "published",
                GradientConvention::Hirshfeld => "hirshfeld",
            },
        );
        report.set(
            "igmh_proatoms",
            "neutral spherically averaged HF SAD in the molecular AO basis",
        );
        report.set(
            "igmh_points",
            self.points.iter().map(|p| p.to_array()).collect::<Vec<_>>(),
        );
        if let Some(shape) = self.shape {
            report.set("igmh_grid_shape", shape);
        }
        report.set(
            "igmh_density",
            output.points.iter().map(|p| p.density).collect::<Vec<_>>(),
        );
        report.set(
            "igmh_density_gradient",
            output
                .points
                .iter()
                .map(|p| p.density_gradient.to_array())
                .collect::<Vec<_>>(),
        );
        report.set(
            "igmh_density_hessian",
            output
                .points
                .iter()
                .map(|p| p.density_hessian)
                .collect::<Vec<_>>(),
        );
        report.set(
            "igmh_signed_density",
            output
                .points
                .iter()
                .map(|p| p.signed_density)
                .collect::<Vec<_>>(),
        );
        report.set(
            "igmh_delta_g",
            output.points.iter().map(|p| p.delta_g).collect::<Vec<_>>(),
        );
        report.set(
            "igmh_delta_g_inter",
            output
                .points
                .iter()
                .map(|p| p.delta_g_inter)
                .collect::<Vec<_>>(),
        );
        report.set(
            "igmh_delta_g_intra",
            output
                .points
                .iter()
                .map(|p| p.delta_g_intra)
                .collect::<Vec<_>>(),
        );
        report.set("igmh_density_residual", output.density_residual);
        report.set("igmh_gradient_residual", output.gradient_residual);
        report.set("igmh_vacuum_points", output.vacuum_points);
        let fragments = self
            .fragments
            .clone()
            .unwrap_or_else(|| vec![0; setup.molecule.len()]);
        report.set("fragments", fragments);
        if self.atom_details {
            report.set(
                "igmh_atom_densities",
                output
                    .points
                    .iter()
                    .map(|p| &p.atom_densities)
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            report.set(
                "igmh_atom_vectors",
                output
                    .points
                    .iter()
                    .map(|p| {
                        p.atom_vectors
                            .iter()
                            .map(|v| v.to_array())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            );
        }
        Ok(())
    }
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let request = Request::parse(controls)?;
    let kind = CollinearPropertyReference::from_controls(controls)?;
    let setup = setup(system, controls)?;
    request.validate(&setup)?;
    controls.finish("igmh")?;
    let reference = run_collinear_property_reference(&setup, kind)?;
    let mut density = reference.scf.density[0].clone();
    for (a, b) in density
        .as_mut_slice()
        .iter_mut()
        .zip(reference.scf.density[1].as_slice())
    {
        *a += b;
    }
    let mut report = reference_report(
        &setup,
        &reference,
        "independent gradient model based on Hirshfeld partition",
    );
    let reference_type = match kind {
        CollinearPropertyReference::Roks => "roks",
        _ if reference.scf.unrestricted => "uks",
        _ => "rks",
    };
    report.set("reference_type", reference_type);
    report.set(
        "scf_orbital_residual",
        reference_orbital_residual(&reference.scf),
    );
    request.append(&mut report, &setup, &density)?;
    Ok(report)
}
